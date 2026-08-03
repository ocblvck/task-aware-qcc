"""R2 — downstream attack-detection accuracy under depolarizing noise.

Builds a quantum-kernel classifier (QSVC: a single precomputed-kernel SVC) from
a *candidate* feature map and measures how well it still detects intrusions on a
small IoT-NID / UNSW-NB15 split, evaluated under a depolarizing density-matrix
channel. This is the "practically useful for intrusion detection on noisy
near-term hardware" half of the task-aware reward.

The classifier protocol mirrors ``run_hardware_kernel.fit_eval_model`` /
``device_noise_validation`` (precomputed Gram, ``class_weight='balanced'``), so
the numbers are comparable to the paper. QVE-style multi-map ensembles are
supported via :func:`downstream_accuracy_ensemble` for later objectives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.svm import SVC

from .kernels import gram_pair


@dataclass
class DownstreamConfig:
    """Configuration for the downstream evaluation in the R2 reward."""

    num_qubits: int = 6
    noise_p1: float = 0.01          # 1q depolarizing strength (0 => exact path)
    noise_p2: Optional[float] = None  # 2q strength (default 10*p1)
    seed: int = 42
    gpu: bool = True


def _fit_eval_single(K_train, K_test, y_train, y_test, seed) -> Tuple[float, float, np.ndarray]:
    svc = SVC(kernel="precomputed", class_weight="balanced",
              probability=False, random_state=seed)
    svc.fit(K_train, y_train)
    pred = svc.predict(K_test)
    acc = float(accuracy_score(y_test, pred))
    mcc = float(matthews_corrcoef(y_test, pred)) if len(np.unique(y_test)) > 1 else 0.0
    return acc, mcc, pred


def downstream_accuracy(
    feature_map,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: DownstreamConfig,
) -> dict:
    """QSVC accuracy/MCC for a single feature map under the configured noise."""
    K_train, K_test = gram_pair(
        feature_map, X_train, X_test,
        p1=cfg.noise_p1, p2=cfg.noise_p2, gpu=cfg.gpu,
    )
    acc, mcc, _ = _fit_eval_single(K_train, K_test, y_train, y_test, cfg.seed)
    return {"accuracy": acc, "mcc": mcc, "noise_p1": cfg.noise_p1}


def downstream_accuracy_ensemble(
    feature_maps: Sequence,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: DownstreamConfig,
    majority_vote: bool = True,
) -> dict:
    """QVE-style hard-majority-vote ensemble over several feature maps (R2 for QVE)."""
    preds: List[np.ndarray] = []
    for fm in feature_maps:
        K_train, K_test = gram_pair(
            fm, X_train, X_test, p1=cfg.noise_p1, p2=cfg.noise_p2, gpu=cfg.gpu,
        )
        _, _, p = _fit_eval_single(K_train, K_test, y_train, y_test, cfg.seed)
        preds.append(p)
    P = np.array(preds)
    if P.shape[0] > 1 and majority_vote:
        y_pred = (P.mean(0) >= 0.5).astype(int)
    else:
        y_pred = P[0]
    acc = float(accuracy_score(y_test, y_pred))
    mcc = float(matthews_corrcoef(y_test, y_pred)) if len(np.unique(y_test)) > 1 else 0.0
    return {"accuracy": acc, "mcc": mcc, "noise_p1": cfg.noise_p1}


def downstream_accuracy_weighted_ensemble(
    feature_maps: Sequence,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: DownstreamConfig,
) -> dict:
    """QWE-style weighted vote; weights = per-map accuracy on an 80/20 internal split.

    Mirrors the paper's QWE: each member's weight is its validation accuracy, so
    stronger encodings dominate. Needs >=3 members for weighting to matter beyond
    a 2-way tie-break.
    """
    from sklearn.model_selection import train_test_split

    strat = y_train if len(np.unique(y_train)) > 1 else None
    Xtr2, Xval, ytr2, yval = train_test_split(
        X_train, y_train, test_size=0.2, random_state=cfg.seed, stratify=strat)

    weights, preds = [], []
    for fm in feature_maps:
        Ktr2, Kval = gram_pair(fm, Xtr2, Xval, p1=cfg.noise_p1, p2=cfg.noise_p2, gpu=cfg.gpu)
        acc_v, _, _ = _fit_eval_single(Ktr2, Kval, ytr2, yval, cfg.seed)
        weights.append(max(acc_v, 1e-6))
        Ktr, Kte = gram_pair(fm, X_train, X_test, p1=cfg.noise_p1, p2=cfg.noise_p2, gpu=cfg.gpu)
        _, _, p = _fit_eval_single(Ktr, Kte, y_train, y_test, cfg.seed)
        preds.append(p)

    w = np.array(weights) / np.sum(weights)
    y_pred = ((w[:, None] * np.array(preds)).sum(0) >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, y_pred))
    mcc = float(matthews_corrcoef(y_test, y_pred)) if len(np.unique(y_test)) > 1 else 0.0
    return {"accuracy": acc, "mcc": mcc, "noise_p1": cfg.noise_p1, "weights": w.tolist()}

