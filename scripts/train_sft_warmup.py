#!/usr/bin/env python
"""SFT warmup: teach Qwen2.5-Coder-3B to emit valid feature-map compression QASM.

Raw Qwen doesn't reliably emit parseable OpenQASM 3.0 feature maps, so the GRPO
task reward stays flat at the invalid penalty. This short LoRA SFT on
prompt->(L3-equivalent QASM) pairs unblocks it. The resulting adapter is then
merged before GRPO via ``train_taskaware_grpo.py --sft-adapter``.

Run (1 GPU):
  CUDA_VISIBLE_DEVICES=0 python scripts/train_sft_warmup.py \
      --base-model Qwen/Qwen2.5-Coder-3B-Instruct --output models/sft_warmup \
      --epochs 4
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
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--output", default="models/sft_warmup")
    ap.add_argument("--qubits", type=int, nargs="+", default=[4, 6])
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=4224,
                    help="Must fit prompt+completion; longest examples are ~4160 tokens")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--compressed-targets", action="store_true",
                    help="Teach a compression prior (full->linear entanglement targets)")
    args = ap.parse_args()

    import torch
    import transformers
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    from taqcc.grpo_integration import build_sft_dataset

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = build_sft_dataset(
        tokenizer, num_qubits_list=args.qubits, repeats=args.repeats,
        compressed_targets=args.compressed_targets,
    )
    # Report token lengths so we size max_length / GRPO max_prompt_length well.
    lens = [len(tokenizer(r["prompt"] + r["completion"])["input_ids"]) for r in ds]
    plens = [len(tokenizer(r["prompt"])["input_ids"]) for r in ds]
    print(f"[sft-data] examples={len(ds)} | prompt+compl tokens "
          f"min/med/max={min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)} | "
          f"prompt tokens max={max(plens)}", flush=True)

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", use_rslora=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    sft_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=torch.cuda.is_available(),
        max_length=args.max_length,
        completion_only_loss=True,   # mask the prompt; train on the QASM answer
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=args.base_model,
        args=sft_config,
        train_dataset=ds,
        peft_config=lora,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"[done] SFT warmup adapter saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
