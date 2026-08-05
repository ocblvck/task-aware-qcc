"""Lightweight, paper-faithful data loading for the task-aware reward.

This is a dependency-light reimplementation of
``quantum-ml-iot-nid/circuit_depth_experiment.py:DataProcessor`` +
``device_noise_validation.load_dataset``. It mirrors the exact preprocessing
pipeline (drop leaking columns -> SelectKBest -> StandardScaler -> PCA ->
MinMaxScaler[0, pi]) but uses **sklearn only** (no cuML), so it imports cleanly
in any Qiskit env and never disturbs the running paper experiments.

For the final paper runs you can swap this for the original loader to get
byte-identical features; for reward evaluation the pipeline is identical.
"""

from __future__ import annotations

import csv
from typing import Tuple
from functools import partial

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

_LEAKING_COLS = {
    "pkSeqID", "id", "Cat", "Sub_Cat", "category", "subcategory", "attack_cat",
}
_TARGET_CANDIDATES = ("Label", "label", "attack", "Attack")


def _detect_delimiter(path: str) -> str:
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as fh:
        sample = fh.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


class DataProcessor:
    """Mirror of the paper DataProcessor (sklearn-only).

    Pipeline (fit on TRAIN only to avoid leakage):
        SelectKBest(mutual_info) -> StandardScaler -> PCA(num_qubits) -> MinMax[0, pi]
    """

    def __init__(self, num_qubits: int, random_seed: int = 42):
        self.num_qubits = num_qubits
        self.random_seed = random_seed
        self.selector = None
        self.pre_pca_scaler = None
        self.pca = None
        self.scaler = None
        self.label_encoder = None

    def prepare_data(self, df: pd.DataFrame, sample_size: int | None = None):
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        drop = [c for c in df.columns if c.startswith("Unnamed")]
        drop += [c for c in _LEAKING_COLS if c in df.columns]
        if drop:
            df = df.drop(columns=sorted(set(drop)))

        target = next((c for c in _TARGET_CANDIDATES if c in df.columns), df.columns[-1])
        X = df.drop(target, axis=1)
        y = df[target]

        cat = X.select_dtypes(include=["object"]).columns.tolist()
        if cat:
            X = X.drop(columns=cat)

        X = X.values.astype(np.float32)
        y = y.values
        if y.dtype == "object":
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        if sample_size and sample_size < len(X):
            rng = np.random.default_rng(self.random_seed)
            uniq, counts = np.unique(y, return_counts=True)
            minority_ratio = counts.min() / counts.max()
            # Guard against losing the minority class on extreme imbalance.
            if len(uniq) == 2 and minority_ratio * sample_size < 10 and counts.min() > 0:
                minc = uniq[np.argmin(counts)]
                mino = np.where(y == minc)[0]
                majo = np.where(y != minc)[0]
                t_min = min(len(mino), max(10, sample_size // 10))
                t_maj = sample_size - t_min
                sel = np.concatenate([
                    rng.choice(mino, t_min, replace=False),
                    rng.choice(majo, t_maj, replace=False),
                ])
                rng.shuffle(sel)
                X, y = X[sel], y[sel]
            else:
                _, X, _, y = train_test_split(
                    X, y, test_size=sample_size / len(X),
                    stratify=y, random_state=self.random_seed,
                )

        X = np.nan_to_num(X, nan=0.0, posinf=np.pi, neginf=0.0)
        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DataProcessor":
        if X.shape[1] > self.num_qubits * 2:
            # Bind a fixed random_state so mutual-information estimation is deterministic.
            score_fn = partial(mutual_info_classif, random_state=self.random_seed)
            self.selector = SelectKBest(
                score_fn, k=min(self.num_qubits * 2, X.shape[1])
            )
            X = self.selector.fit_transform(X, y)
        self.pre_pca_scaler = StandardScaler()
        X = self.pre_pca_scaler.fit_transform(X)
        if X.shape[1] > self.num_qubits:
            self.pca = PCA(n_components=self.num_qubits, random_state=self.random_seed)
            X = self.pca.fit_transform(X)
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selector is not None:
            X = self.selector.transform(X)
        X = self.pre_pca_scaler.transform(X)
        if self.pca is not None:
            X = self.pca.transform(X)
        elif X.shape[1] < self.num_qubits:
            X = np.hstack([X, np.zeros((X.shape[0], self.num_qubits - X.shape[1]))])
        X = self.scaler.transform(X)
        return np.nan_to_num(X, nan=0.0, posinf=np.pi, neginf=0.0).astype(np.float32)


def load_split(
    dataset_path: str,
    num_qubits: int,
    train_size: int,
    test_size: int,
    pool_size: int = 5000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a small, balanced, leakage-safe train/test split ready for encoding.

    Returns angle-encoded ``X_train, X_test, y_train, y_test`` in ``[0, pi]``.
    """
    path = str(dataset_path)
    df = pd.read_csv(path, sep=_detect_delimiter(path), low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    proc = DataProcessor(num_qubits=num_qubits, random_seed=seed)
    X, y = proc.prepare_data(df, sample_size=pool_size)

    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=test_size, train_size=train_size,
        random_state=seed, stratify=y,
    )
    enc = DataProcessor(num_qubits=num_qubits, random_seed=seed).fit(Xtr_raw, ytr)
    return enc.transform(Xtr_raw), enc.transform(Xte_raw), ytr, yte


def load_split_multiclass(
    dataset_path: str,
    num_qubits: int,
    train_size: int,
    test_size: int,
    target: str = "Cat",
    pool_size: int = 5000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """DCAS-style multi-class split (IoTID20 5-class via the ``Cat`` column).

    Drops binary/leaking label columns, keeps ``target`` as the label, then applies
    the same SelectKBest -> StandardScaler -> PCA -> MinMax[0,pi] pipeline fit on
    the training partition only. Returns angle-encoded train/test splits.
    """
    from sklearn.preprocessing import LabelEncoder

    path = str(dataset_path)
    df = pd.read_csv(path, sep=_detect_delimiter(path), low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not in dataset")

    y = LabelEncoder().fit_transform(df[target].values)
    drop = [c for c in df.columns if c.startswith("Unnamed")]
    drop += [c for c in ("pkSeqID", "id", "Label", "label", "Cat", "Sub_Cat",
                          "category", "subcategory", "attack_cat", "attack", "Attack")
             if c in df.columns]
    X = df.drop(columns=sorted(set(drop)))
    X = X.drop(columns=X.select_dtypes(include=["object"]).columns.tolist())
    X = np.nan_to_num(X.values.astype(np.float32), nan=0.0, posinf=np.pi, neginf=0.0)

    # Stratified subsample to pool_size for tractable kernel construction.
    if pool_size and pool_size < len(X):
        _, X, _, y = train_test_split(X, y, test_size=pool_size / len(X),
                                      stratify=y, random_state=seed)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=test_size, train_size=train_size,
        random_state=seed, stratify=y,
    )
    enc = DataProcessor(num_qubits=num_qubits, random_seed=seed).fit(Xtr_raw, ytr)
    return enc.transform(Xtr_raw), enc.transform(Xte_raw), ytr, yte
