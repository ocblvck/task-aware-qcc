"""Task-Aware, Noise-Aware RL for Quantum Circuit Compression (taqcc).

This package glues two existing projects on this machine:

  * ``quantum-cirq-opt``: Qwen2.5-Coder-3B + LoRA + GRPO that emits/compresses
    OpenQASM circuits with an 11-component *proxy* reward (``src/qcc``).
  * ``quantum-ml-iot-nid``: QSVC/QVE/QWE quantum-kernel IoT intrusion detection
    under depolarizing noise (feature maps, exact + noisy kernels, IoT-NID / UNSW).

The novel contribution lives in :mod:`taqcc.reward`: a **two-part, task-aware**
GRPO reward that replaces (or augments) the structural proxy reward with

  R1 = exact equivalence verification on circuits small enough to simulate, and
  R2 = downstream attack-detection accuracy of a QSVC/QVE kernel built from the
       compressed feature map, evaluated under depolarizing noise.

Nothing here mutates either source project; we import/port the minimal reusable
pieces so the source experiments (incl. the running hardware ladder) are untouched.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .feature_maps import make_feature_map, circuit_metrics
from .equivalence import equivalence_score
from .downstream import downstream_accuracy, DownstreamConfig
from .reward import (
    TaskAwareRewardConfig,
    task_aware_reward,
    score_candidate,
)

__all__ = [
    "make_feature_map",
    "circuit_metrics",
    "equivalence_score",
    "downstream_accuracy",
    "DownstreamConfig",
    "TaskAwareRewardConfig",
    "task_aware_reward",
    "score_candidate",
]
