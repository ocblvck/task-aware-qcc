"""Wire the task-aware reward into ``quantum-cirq-opt``'s GRPO trainer.

This module does the plumbing to train Qwen2.5-Coder-3B + LoRA + GRPO with the
two-part task-aware reward, **without modifying either source project**:

  * :func:`build_feature_map_dataset`: turns the paper's feature maps into a
    GRPO dataset of *compression tasks* (prompt asks to shorten a given QASM
    feature map; ``solution`` carries the original QASM for the reward).
  * :func:`make_grpo_reward`: a TRL reward ``fn(prompts, completions, **kwargs)``
    that reads the per-sample original feature map from ``kwargs['solution']``,
    caches a :class:`~taqcc.reward.TaskContext` per unique original (with baseline
    accuracy), and scores each completion with :func:`taqcc.reward.score_candidate`.
  * :class:`TaskAwareGRPOTrainer`: subclass of ``ComplexityAwareGRPOTrainer``
    that swaps in the task-aware reward (optionally blended with cheap
    format/syntax rewards for training stability).

All heavy imports (``qcc`` / trl / torch) are lazy so this module imports fine in
the reward-only env; only :class:`TaskAwareGRPOTrainer` needs the GRPO env.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile

from .data import load_split
from .downstream import DownstreamConfig
from .feature_maps import MODEL_MAPS, circuit_metrics, make_feature_map
from .qasm_adapter import parse_candidate
from .reward import TaskAwareRewardConfig, TaskContext, score_candidate

# Make the sibling source project importable (for qcc helpers / trainer).
_QCC_SRC = "/home/chibuike/quantum-cirq-opt/src"
if Path(_QCC_SRC).exists() and _QCC_SRC not in sys.path:
    sys.path.append(_QCC_SRC)

_BASIS = ["u", "cx", "rz", "sx", "x"]

_TASK_SYSTEM_PROMPT = (
    "You are an expert quantum compiler. You compress parameterized quantum "
    "feature-map circuits used by a quantum-kernel intrusion-detection classifier. "
    "Given an OpenQASM 3.0 feature map with input parameters, output an equivalent "
    "but SHORTER circuit (fewer two-qubit gates and lower depth) that keeps exactly "
    "the same input parameters and preserves the encoded quantum states. "
    "Return only valid OpenQASM 3.0 code."
)


# --------------------------------------------------------------------------- #
# QASM export of the original feature maps (the compression targets).
# --------------------------------------------------------------------------- #
def feature_map_to_qasm(fm: QuantumCircuit) -> str:
    """Decompose to the paper basis and export gate-level OpenQASM 3.0."""
    from qiskit import qasm3

    decomposed = transpile(fm, basis_gates=_BASIS, optimization_level=0)
    return qasm3.dumps(decomposed)


def _compression_prompt(qasm: str, num_qubits: int, map_type: str) -> str:
    return (
        f"Compress the following {num_qubits}-qubit '{map_type}' quantum feature "
        f"map. Produce an OpenQASM 3.0 circuit with the SAME {num_qubits} input "
        f"parameters that prepares the same encoded state with fewer two-qubit "
        f"gates and lower depth.\n\nOriginal circuit:\n```qasm\n{qasm}\n```\n\n"
        f"Return only the compressed OpenQASM 3.0 code."
    )


def build_feature_map_dataset(
    tokenizer,
    num_qubits: int = 6,
    specs: Optional[Sequence[Tuple[str, int, str]]] = None,
    repeats: int = 8,
):
    """Build a GRPO ``DatasetDict`` of feature-map compression tasks.

    ``specs`` is a list of ``(map_type, reps, entanglement)``; defaults to the
    union of maps used by QSVC/QVE/QWE. Each spec is repeated ``repeats`` times
    (GRPO samples multiple completions per prompt anyway; repeats give the policy
    more optimizer steps over the same targets).
    """
    from datasets import Dataset, DatasetDict

    if specs is None:
        seen, specs = set(), []
        for maps in MODEL_MAPS.values():
            for map_type, reps in maps:
                for ent in ("full", "linear"):
                    key = (map_type, reps, ent)
                    if key not in seen:
                        seen.add(key)
                        specs.append(key)

    rows = []
    for (map_type, reps, ent) in specs:
        fm = make_feature_map(num_qubits, map_type, reps, ent)
        qasm = feature_map_to_qasm(fm)
        prompt_text = _compression_prompt(qasm, num_qubits, map_type)
        formatted = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": _TASK_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for _ in range(repeats):
            rows.append({
                "prompt": formatted,
                "solution": qasm,          # original QASM -> reward reference
                "num_qubits": num_qubits,
                "map_type": map_type,
                "reps": reps,
                "entanglement": ent,
            })

    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=max(1, len(rows) // 20), seed=42)
    return DatasetDict(train=split["train"], test=split["test"])


# --------------------------------------------------------------------------- #
# SFT warmup dataset: teach the policy to emit valid, equivalent, gate-level
# feature-map QASM for the compression prompt (so the GRPO task reward can fire).
# --------------------------------------------------------------------------- #
def _l3_equivalent_target(fm: QuantumCircuit) -> Optional[str]:
    """Qiskit L3-optimized, provably-equivalent QASM target for ``fm``.

    Returns ``None`` if L3 is not gate-equivalent (defensive; should not happen)
    or is not smaller-or-equal than the input.
    """
    from qiskit import qasm3

    from .equivalence import equivalence_score
    from .feature_maps import circuit_metrics

    target = transpile(fm, basis_gates=_BASIS, optimization_level=3, seed_transpiler=42)
    eq = equivalence_score(fm, target, num_samples=6, max_exact_qubits=14, seed=0)
    if eq is None or eq < 0.999:
        return None
    # Only keep targets that are no larger than the source (compression demo).
    if circuit_metrics(target)["two_qubit"] > circuit_metrics(fm)["two_qubit"]:
        return None
    return qasm3.dumps(target)


def _compressed_target(nq: int, map_type: str, reps: int, ent: str):
    """A genuinely SMALLER target circuit that teaches a compression prior.

    For full-entanglement sources we target the linear-entanglement version of
    the same map (~half the two-qubit gates), a legitimate cheaper kernel that
    the paper finds is often *more* noise-robust. For sources that are already
    minimal (linear / single-qubit Z), we fall back to the L3-equivalent target.
    Returns (target_qasm, is_compressed).
    """
    if ent == "full":
        linear = make_feature_map(nq, map_type, reps, "linear")
        src2q = circuit_metrics(make_feature_map(nq, map_type, reps, ent))["two_qubit"]
        tgt2q = circuit_metrics(linear)["two_qubit"]
        if tgt2q < src2q:
            return feature_map_to_qasm(linear), True
    return _l3_equivalent_target(make_feature_map(nq, map_type, reps, ent)), False


def build_sft_dataset(
    tokenizer,
    num_qubits_list: Sequence[int] = (4, 6),
    specs: Optional[Sequence[Tuple[str, int, str]]] = None,
    repeats: int = 4,
    compressed_targets: bool = False,
):
    """Build a prompt→completion SFT dataset of feature-map compression examples.

    Prompts use the *identical* system+user format as
    :func:`build_feature_map_dataset`, so the SFT warmup transfers directly to
    GRPO. With ``compressed_targets=False`` completions are L3-optimized,
    provably-equivalent QASM (valid emission only). With ``compressed_targets=True``
    full-entanglement sources map to their smaller linear-entanglement version,
    giving the policy a real *compression prior* for GRPO to refine.
    """
    from datasets import Dataset
    from qiskit import qasm3

    if specs is None:
        seen, specs = set(), []
        for maps in MODEL_MAPS.values():
            for map_type, reps in maps:
                for ent in ("full", "linear"):
                    key = (map_type, reps, ent)
                    if key not in seen:
                        seen.add(key)
                        specs.append(key)

    rows = []
    for nq in num_qubits_list:
        for (map_type, reps, ent) in specs:
            fm = make_feature_map(nq, map_type, reps, ent)
            src_qasm = feature_map_to_qasm(fm)
            if compressed_targets:
                tgt_qasm, _is_comp = _compressed_target(nq, map_type, reps, ent)
            else:
                tgt_qasm = _l3_equivalent_target(fm)
            if tgt_qasm is None:
                continue
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": _TASK_SYSTEM_PROMPT},
                    {"role": "user", "content": _compression_prompt(src_qasm, nq, map_type)},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for _ in range(repeats):
                rows.append({
                    "prompt": prompt,
                    "completion": tgt_qasm,
                    "num_qubits": nq,
                    "map_type": map_type,
                })

    return Dataset.from_list(rows)




# --------------------------------------------------------------------------- #
# Reward factory: TRL-compatible, per-sample original from kwargs, cached context.
# --------------------------------------------------------------------------- #
def _hash(qasm: str) -> str:
    return hashlib.sha1(qasm.encode("utf-8")).hexdigest()[:16]


def make_grpo_reward(
    dataset_path: str,
    num_qubits: int = 6,
    train_size: int = 16,
    test_size: int = 8,
    pool_size: int = 4000,
    noise_p1: float = 0.01,
    seed: int = 42,
    reward_cfg: Optional[TaskAwareRewardConfig] = None,
    gpu: bool = True,
) -> Callable:
    """Return a TRL reward bound to a fixed IoT/UNSW split and cached contexts."""
    reward_cfg = reward_cfg or TaskAwareRewardConfig()
    X_tr, X_te, y_tr, y_te = load_split(
        dataset_path, num_qubits, train_size, test_size, pool_size=pool_size, seed=seed
    )
    dcfg = DownstreamConfig(num_qubits=num_qubits, noise_p1=noise_p1, seed=seed, gpu=gpu)
    ctx_cache: Dict[str, TaskContext] = {}

    def _context_for(original_qasm: str) -> Optional[TaskContext]:
        key = _hash(original_qasm)
        if key not in ctx_cache:
            orig = parse_candidate(original_qasm)
            if orig is None or orig.num_qubits != num_qubits:
                return None
            ctx_cache[key] = TaskContext(
                original=orig, X_train=X_tr, y_train=y_tr,
                X_test=X_te, y_test=y_te, downstream=dcfg,
            )
        return ctx_cache[key]

    def task_aware_grpo_reward(prompts=None, completions=None, **kwargs) -> List[float]:
        comps = completions or []
        solutions = kwargs.get("solution") or [None] * len(comps)
        rewards: List[float] = []
        for comp, orig_qasm in zip(comps, solutions):
            text = comp if isinstance(comp, str) else _completion_text(comp)
            ctx = _context_for(orig_qasm) if orig_qasm else None
            if ctx is None:
                rewards.append(reward_cfg.invalid_penalty)
                continue
            rewards.append(score_candidate(text, ctx, reward_cfg)["reward"])
        return rewards

    task_aware_grpo_reward.__name__ = "task_aware_grpo_reward"
    return task_aware_grpo_reward


def _completion_text(completion) -> str:
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[-1].get("content", "")
    return str(completion)


# --------------------------------------------------------------------------- #
# Trainer subclass. Swaps in the task-aware reward.
# --------------------------------------------------------------------------- #
def make_task_aware_trainer_class():
    """Return a ``TaskAwareGRPOTrainer`` subclass (lazy import of the GRPO env)."""
    from qcc.grpo_trainer import ComplexityAwareGRPOTrainer  # requires trl/torch
    from qcc.grpo_rewards import format_reward, syntax_reward

    class TaskAwareGRPOTrainer(ComplexityAwareGRPOTrainer):
        """GRPO trainer whose reward is the task-aware R1+R2 objective.

        Set ``self.task_reward`` (from :func:`make_grpo_reward`) before ``train``.
        ``blend_syntax=True`` prepends cheap format+syntax rewards for stability;
        remember to set ``config.reward_weights`` to match the reward count.
        """

        def __init__(self, *args, task_reward: Callable, blend_syntax: bool = True, **kw):
            super().__init__(*args, **kw)
            self.task_reward = task_reward
            self.blend_syntax = blend_syntax
            self._format_reward = format_reward
            self._syntax_reward = syntax_reward

        def get_reward_functions(self):
            if self.blend_syntax:
                return [self._format_reward, self._syntax_reward, self.task_reward]
            return [self.task_reward]

    return TaskAwareGRPOTrainer
