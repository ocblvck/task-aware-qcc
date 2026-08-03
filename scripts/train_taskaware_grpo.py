#!/usr/bin/env python
"""Launch GRPO training with the task-aware, noise-aware reward.

Wires ``taqcc`` into ``quantum-cirq-opt``'s ``ComplexityAwareGRPOTrainer`` without
modifying either source project. Requires the GRPO env (torch-CUDA + trl + peft +
bitsandbytes) AND the quantum stack (qiskit + qiskit-aer + sklearn).

Smoke test (few steps, tiny data — verifies the whole loop end to end):
  python scripts/train_taskaware_grpo.py --smoke \
      --dataset UNSW_NB15.csv --data-dir /home/chibuike/quantum-ml-iot-nid

Real run (single node, 3x A6000 via accelerate):
  accelerate launch --num_processes 3 scripts/train_taskaware_grpo.py \
      --dataset UNSW_NB15.csv --sft-adapter /home/chibuike/quantum-cirq-opt/models/sft_coder3b \
      --num-qubits 6 --train-size 24 --test-size 12 --noise-p1 0.01 \
      --max-steps 500 --output models/taqcc_grpo_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
for _p in ("/home/chibuike/quantum-cirq-opt/src",):
    if Path(_p).exists() and _p not in sys.path:
        sys.path.append(_p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="UNSW_NB15.csv")
    ap.add_argument("--data-dir", default="/home/chibuike/quantum-ml-iot-nid")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--sft-adapter", default=None,
                    help="Path to SFT LoRA adapter to merge before RL (recommended)")
    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--train-size", type=int, default=16)
    ap.add_argument("--test-size", type=int, default=8)
    ap.add_argument("--pool-size", type=int, default=4000)
    ap.add_argument("--noise-p1", type=float, default=0.01)
    ap.add_argument("--gate-mode", default="and", choices=["and", "or"])
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="GRPO sampling temperature (raise >1 for more exploration)")
    ap.add_argument("--lr", type=float, default=1e-6,
                    help="Learning rate (default 1e-6; raise cautiously to move off a peaked prior)")
    ap.add_argument("--max-prompt-length", type=int, default=2304,
                    help="Must fit the original QASM in the prompt (~2164 tokens)")
    ap.add_argument("--max-completion-length", type=int, default=2048)
    ap.add_argument("--task-weight", type=float, default=1.0)
    ap.add_argument("--no-blend-syntax", action="store_true",
                    help="Use ONLY the task-aware reward (no format/syntax stabilizers)")
    ap.add_argument("--no-gpu-sim", action="store_true",
                    help="Run the reward's density-matrix sim on CPU Aer")
    ap.add_argument("--w-comp", type=float, default=None, help="compression weight (default 3.0)")
    ap.add_argument("--w-equiv", type=float, default=None, help="equivalence weight (default 0.3)")
    ap.add_argument("--w-util", type=float, default=None, help="utility weight (default 0.5)")
    ap.add_argument("--output", default="models/taqcc_grpo_v1")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny end-to-end validation (few steps, small maps)")
    args = ap.parse_args()

    from qcc.grpo_trainer import GRPOTrainingConfig
    from taqcc.reward import TaskAwareRewardConfig
    from taqcc.grpo_integration import (
        build_feature_map_dataset, make_grpo_reward, make_task_aware_trainer_class,
    )

    if args.smoke:
        args.repeats = 2
        args.max_steps = 2
        args.num_generations = 2
        args.batch_size = 1
        args.train_size = min(args.train_size, 10)
        args.test_size = min(args.test_size, 6)

    blend = not args.no_blend_syntax
    # reward_weights MUST match the number of reward functions returned by
    # get_reward_functions(): [format, syntax, task] when blending, else [task].
    reward_weights = [0.3, 0.5, args.task_weight] if blend else [args.task_weight]

    config = GRPOTrainingConfig(
        base_model_name=args.base_model,
        sft_adapter_path=args.sft_adapter,
        output_dir=args.output,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        learning_rate=args.lr,
        reward_weights=reward_weights,
        # anti-mode-collapse defaults (dapo, beta=0) come from the config.
    )

    reward_cfg = TaskAwareRewardConfig(gate_mode=args.gate_mode)
    if args.w_comp is not None:
        reward_cfg.w_comp = args.w_comp
    if args.w_equiv is not None:
        reward_cfg.w_equiv = args.w_equiv
    if args.w_util is not None:
        reward_cfg.w_util = args.w_util
    print(f"[reward] gate_mode={reward_cfg.gate_mode} w_comp={reward_cfg.w_comp} "
          f"w_equiv={reward_cfg.w_equiv} w_util={reward_cfg.w_util}", flush=True)
    task_reward = make_grpo_reward(
        dataset_path=str(Path(args.data_dir) / args.dataset),
        num_qubits=args.num_qubits,
        train_size=args.train_size,
        test_size=args.test_size,
        pool_size=args.pool_size,
        noise_p1=args.noise_p1,
        reward_cfg=reward_cfg,
        gpu=not args.no_gpu_sim,
    )

    TaskAwareGRPOTrainer = make_task_aware_trainer_class()
    trainer = TaskAwareGRPOTrainer(
        config=config, task_reward=task_reward, blend_syntax=blend,
    )
    trainer.setup()  # loads tokenizer (needed to build the dataset)

    dataset = build_feature_map_dataset(
        trainer.tokenizer, num_qubits=args.num_qubits, repeats=args.repeats,
    )
    print(f"[dataset] train={len(dataset['train'])} test={len(dataset['test'])} "
          f"reward_funcs={'format+syntax+task' if blend else 'task-only'} "
          f"weights={reward_weights}", flush=True)

    trainer.train(dataset=dataset)
    print(f"[done] model saved under {args.output}", flush=True)


if __name__ == "__main__":
    main()
