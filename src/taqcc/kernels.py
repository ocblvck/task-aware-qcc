"""Quantum-kernel evaluation: exact statevector + noisy density-matrix Gram.

Mirrors the kernel math used across ``quantum-ml-iot-nid`` but in a compact,
self-contained form:

  * Exact path, ``ideal_gram`` / ``ideal_rect`` (statevector inner products),
    copied from ``run_hardware_kernel.py``.
  * Noisy path, depolarizing density-matrix kernel using the measurement-free
    Hilbert-Schmidt overlap  K(x, z) = Tr(rho(x) rho(z))  exactly as the paper's
    ``NoisyFidelityKernel`` defines it (so reward numbers are comparable).

The noisy path uses ``AerSimulator(method='density_matrix')`` with a uniform
depolarizing channel on 1q/2q gates. GPU is used automatically when the installed
qiskit-aer is a GPU build (``device='GPU'`` is attempted, falls back to CPU).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

_BASIS = ["u", "cx", "rz", "sx", "x"]


# --------------------------------------------------------------------------- #
# Exact (noiseless) statevector kernel, the reference rung.
# --------------------------------------------------------------------------- #
def _statevectors(fm: QuantumCircuit, X: np.ndarray):
    return [Statevector(fm.assign_parameters(x)) for x in X]


def ideal_gram(fm: QuantumCircuit, X: np.ndarray) -> np.ndarray:
    svs = _statevectors(fm, X)
    n = len(svs)
    K = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            K[i, j] = K[j, i] = abs(svs[i].inner(svs[j])) ** 2
    return K


def ideal_rect(fm: QuantumCircuit, X_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    sa = _statevectors(fm, X_test)
    sb = _statevectors(fm, X_train)
    return np.array([[abs(s1.inner(s2)) ** 2 for s2 in sb] for s1 in sa])


# --------------------------------------------------------------------------- #
# Depolarizing noise model (uniform), matching the main-study noise sweep knob.
# --------------------------------------------------------------------------- #
def depolarizing_noise_model(p1: float, p2: Optional[float] = None) -> NoiseModel:
    """Uniform depolarizing channel: ``p1`` on 1q gates, ``p2`` (default 10*p1) on 2q."""
    if p2 is None:
        p2 = min(1.0, 10.0 * p1)
    nm = NoiseModel()
    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["u", "rz", "sx", "x"])
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return nm


@lru_cache(maxsize=4)
def _density_sim(noise_key, gpu: bool):
    p1, p2 = noise_key
    nm = depolarizing_noise_model(p1, p2) if p1 or p2 else None
    kwargs = dict(method="density_matrix", noise_model=nm)
    if gpu:
        try:
            sim = AerSimulator(device="GPU", **kwargs)
            return sim
        except Exception:
            pass
    return AerSimulator(**kwargs)


def _density_matrices(fm: QuantumCircuit, X: np.ndarray, sim: AerSimulator):
    """Run each encoded circuit once and save its (noisy) density matrix."""
    rhos = []
    for x in X:
        qc = fm.assign_parameters(x)
        tqc = transpile(qc, basis_gates=_BASIS, optimization_level=0)
        tqc.save_density_matrix()
        res = sim.run(tqc).result()
        rhos.append(np.asarray(res.data(0)["density_matrix"]))
    return rhos


def _hs_overlap(ra: np.ndarray, rb: np.ndarray) -> float:
    # K = Tr(rho_a rho_b), real, clipped to [0, 1].
    val = float(np.real(np.trace(ra @ rb)))
    return min(1.0, max(0.0, val))


def noisy_gram(fm, X, p1, p2=None, gpu=True) -> np.ndarray:
    sim = _density_sim((p1, p2), gpu)
    rhos = _density_matrices(fm, X, sim)
    n = len(rhos)
    K = np.empty((n, n))
    for i in range(n):
        K[i, i] = _hs_overlap(rhos[i], rhos[i])
        for j in range(i + 1, n):
            K[i, j] = K[j, i] = _hs_overlap(rhos[i], rhos[j])
    return K


def noisy_rect(fm, X_test, X_train, p1, p2=None, gpu=True) -> np.ndarray:
    sim = _density_sim((p1, p2), gpu)
    ra = _density_matrices(fm, X_test, sim)
    rb = _density_matrices(fm, X_train, sim)
    return np.array([[_hs_overlap(a, b) for b in rb] for a in ra])


def gram_pair(fm, X_train, X_test, p1=0.0, p2=None, gpu=True):
    """Return (K_train, K_test_rect) for ``fm`` at depolarizing strength ``p1``.

    ``p1 == 0`` uses the exact statevector path (fast, exact); otherwise the
    density-matrix path.
    """
    if not p1 and not p2:
        return ideal_gram(fm, X_train), ideal_rect(fm, X_test, X_train)
    return (
        noisy_gram(fm, X_train, p1, p2, gpu),
        noisy_rect(fm, X_test, X_train, p1, p2, gpu),
    )
