"""Task-aware, noise-aware reward for GRPO circuit compression.

This is the scientific core of the project. It replaces (or augments) the
11-component *structural proxy* reward in ``qcc.grpo_rewards`` with a two-part,
**task-aware** signal:

    R1 (correctness)  exact equivalence on circuits small enough to simulate
    R2 (utility)      downstream QSVC/QVE attack-detection accuracy under
                      depolarizing noise, retained relative to the original map

and gates the *compression* gain by both, so that shrinking a circuit is only
rewarded when the result is **provably correct AND practically useful**:

    reward = w_valid * valid
           + w_equiv * equiv                      # reward provable correctness
           + w_util  * utility                    # reward retained accuracy
           + w_comp  * compression_gain * gate     # reward shrinkage, gated

    gate    = correctness * utility   in [0, 1]
    utility = clip(cand_acc / orig_acc, 0, util_cap)

Penalties are deliberately SOFT (no -1.0 cliffs): the ``quantum-cirq-opt``
project documents that hard penalties + high LR trigger GRPO mode collapse.

The exported :func:`task_aware_reward` matches the TRL reward signature
``fn(prompts, completions, **kwargs) -> list[float]`` via :func:`make_reward_fn`,
so it can be dropped straight into ``ComplexityAwareGRPOTrainer``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from qiskit import QuantumCircuit

from .downstream import DownstreamConfig, downstream_accuracy
from .equivalence import equivalence_score
from .feature_maps import circuit_metrics
from .qasm_adapter import parse_candidate


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class TaskAwareRewardConfig:
    # weights — COMPRESSION-FIRST (task-aware) defaults.
    # Rationale: with a large w_equiv, an equivalent *reproduction* of the source
    # is already a safe high-reward optimum, and real compression of fixed-2q
    # feature maps requires sacrificing equivalence -> the model never compresses.
    # So equivalence is only a small bonus; the dominant driver is
    # compression x retained-accuracy.
    w_valid: float = 0.2
    w_equiv: float = 0.3
    w_util: float = 0.5
    w_comp: float = 3.0
    # soft penalties (no hard cliffs -> avoids GRPO mode collapse)
    invalid_penalty: float = -0.3
    # compression cost model: cost = depth + two_qubit_weight * two_qubit_gates
    two_qubit_weight: float = 2.0
    # utility shaping
    util_cap: float = 1.1          # retention is allowed to slightly exceed 1
    util_floor_acc: float = 0.5    # below this absolute acc, utility -> 0 (chance level)
    # gate semantics for the compression bonus:
    #   "and" (strict)  -> gate = correctness * utility
    #                      shrink only rewarded when BOTH equivalent AND useful.
    #   "or"  (task-aware) -> gate = max(correctness, utility)
    #                      shrink rewarded if EITHER provably equivalent OR it
    #                      retains downstream accuracy (a *better* kernel counts).
    gate_mode: str = "or"
    # equivalence sampling
    equiv_samples: int = 8
    max_exact_qubits: int = 12
    # final clipping
    reward_min: float = -1.0
    reward_max: float = 3.0


# --------------------------------------------------------------------------- #
# Task context — holds the data + the original (reference) feature map and its
# pre-computed baseline accuracy, so per-candidate scoring is cheap-ish.
# --------------------------------------------------------------------------- #
@dataclass
class TaskContext:
    original: QuantumCircuit
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    downstream: DownstreamConfig
    baseline_accuracy: Optional[float] = None

    def ensure_baseline(self) -> float:
        if self.baseline_accuracy is None:
            res = downstream_accuracy(
                self.original, self.X_train, self.y_train,
                self.X_test, self.y_test, self.downstream,
            )
            self.baseline_accuracy = res["accuracy"]
        return self.baseline_accuracy


# --------------------------------------------------------------------------- #
# Single-candidate scoring (also the unit the pilot prints)
# --------------------------------------------------------------------------- #
def _compression_gain(original: QuantumCircuit, candidate: QuantumCircuit, w2q: float) -> float:
    mo, mc = circuit_metrics(original), circuit_metrics(candidate)
    cost_o = mo["depth"] + w2q * mo["two_qubit"]
    cost_c = mc["depth"] + w2q * mc["two_qubit"]
    if cost_o <= 0:
        return 0.0
    return float((cost_o - cost_c) / cost_o)  # >0 means smaller; may be negative


def score_candidate(
    candidate: Union[str, QuantumCircuit, None],
    ctx: TaskContext,
    cfg: TaskAwareRewardConfig,
) -> dict:
    """Return the full reward breakdown for one candidate circuit."""
    circ = parse_candidate(candidate) if isinstance(candidate, str) else candidate
    out = {
        "valid": 0.0, "equiv": None, "candidate_accuracy": None,
        "utility": 0.0, "compression_gain": 0.0, "gate": 0.0,
        "reward": cfg.invalid_penalty,
    }
    if circ is None or circ.num_qubits != ctx.original.num_qubits:
        return out
    # A usable feature map must encode exactly the data features; a different
    # parameter count can't be bound to the data (and would let the policy hack
    # the reward by dropping encodings).
    if circ.num_parameters != ctx.X_train.shape[1]:
        return out
    out["valid"] = 1.0

    # R1 — exact equivalence (None if too large to verify).
    equiv = equivalence_score(
        ctx.original, circ,
        num_samples=cfg.equiv_samples,
        max_exact_qubits=cfg.max_exact_qubits,
        seed=ctx.downstream.seed,
    )
    out["equiv"] = equiv

    # R2 — downstream accuracy retention under noise.
    baseline = ctx.ensure_baseline()
    try:
        res = downstream_accuracy(
            circ, ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test, ctx.downstream
        )
    except Exception:
        out["valid"] = 0.0
        return out
    acc = res["accuracy"]
    out["candidate_accuracy"] = acc
    retention = acc / max(baseline, 1e-6)
    # Absolute-accuracy floor: collapsing to chance kills utility regardless.
    abs_factor = np.clip(
        (acc - cfg.util_floor_acc) / max(1.0 - cfg.util_floor_acc, 1e-6), 0.0, 1.0
    )
    utility = float(np.clip(retention, 0.0, cfg.util_cap) * abs_factor)
    out["utility"] = utility

    # Compression gain (shrinkage), gated by correctness * utility.
    comp = _compression_gain(ctx.original, circ, cfg.two_qubit_weight)
    out["compression_gain"] = comp
    correctness = equiv if equiv is not None else float(np.clip(retention, 0.0, 1.0))
    c, u = float(np.clip(correctness, 0.0, 1.0)), float(np.clip(utility, 0.0, 1.0))
    gate = max(c, u) if cfg.gate_mode == "or" else c * u
    out["gate"] = gate

    equiv_term = (equiv if equiv is not None else 0.0)
    reward = (
        cfg.w_valid * 1.0
        + cfg.w_equiv * equiv_term
        + cfg.w_util * utility
        + cfg.w_comp * comp * gate
    )
    out["reward"] = float(np.clip(reward, cfg.reward_min, cfg.reward_max))
    return out


# --------------------------------------------------------------------------- #
# TRL-compatible batch reward
# --------------------------------------------------------------------------- #
def make_reward_fn(ctx: TaskContext, cfg: Optional[TaskAwareRewardConfig] = None):
    """Bind a :class:`TaskContext` -> a TRL reward ``fn(prompts, completions, **kwargs)``."""
    cfg = cfg or TaskAwareRewardConfig()

    def _reward(prompts=None, completions=None, **kwargs) -> List[float]:
        comps = completions or []
        rewards = []
        for c in comps:
            text = c if isinstance(c, str) else _completion_to_text(c)
            rewards.append(score_candidate(text, ctx, cfg)["reward"])
        return rewards

    _reward.__name__ = "task_aware_reward"
    return _reward


def _completion_to_text(completion) -> str:
    # TRL may pass chat-style [{"role":..,"content":..}] completions.
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[-1].get("content", "")
    return str(completion)


def task_aware_reward(prompts=None, completions=None, **kwargs) -> List[float]:
    """Module-level TRL entry point. Requires a bound context via ``kwargs['ctx']``.

    Prefer :func:`make_reward_fn` for trainer wiring; this exists so the symbol is
    importable/standalone.
    """
    ctx: TaskContext = kwargs["ctx"]
    cfg: TaskAwareRewardConfig = kwargs.get("cfg") or TaskAwareRewardConfig()
    return make_reward_fn(ctx, cfg)(prompts, completions)
