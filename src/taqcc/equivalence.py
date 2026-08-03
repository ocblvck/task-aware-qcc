"""R1 — exact equivalence verification for circuits small enough to simulate.

For a *parameterized* feature map, the only thing the downstream quantum kernel
sees is the encoded state ``|psi(x)> = U(x)|0>`` (because the fidelity kernel is
``K(x, z) = |<psi(x)|psi(z)>|^2``). Two feature maps therefore induce the *same*
kernel iff their encoded states coincide up to a global phase for every ``x`` —
i.e. state fidelity ``= 1``.

We verify this by sampling several random parameter assignments in ``[0, pi]^n``
and averaging the state fidelity ``|<orig|cand>|^2``. This is exact (statevector,
no shots) and is only attempted when ``num_qubits <= max_exact_qubits`` — the
"subset of circuits small enough to simulate exactly" in the proposal.

Returns a value in ``[0, 1]`` (1.0 = provably kernel-equivalent on the sample),
or ``None`` when the circuit is too large to verify exactly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def _ordered_parameters(circuit: QuantumCircuit):
    # Qiskit's ``circuit.parameters`` is already correctly sorted (ParameterVector
    # elements sort numerically, so x[2] < x[10]); use it directly rather than a
    # naive string sort which would misorder double-digit indices.
    return list(circuit.parameters)


def _bind(circuit: QuantumCircuit, values: np.ndarray) -> QuantumCircuit:
    params = _ordered_parameters(circuit)
    return circuit.assign_parameters(dict(zip(params, values)))


def equivalence_score(
    original: QuantumCircuit,
    candidate: QuantumCircuit,
    num_samples: int = 8,
    max_exact_qubits: int = 12,
    seed: int = 0,
) -> Optional[float]:
    """Mean state fidelity between original and candidate over random params.

    ``None`` => not verifiable exactly (too many qubits, or shape mismatch).
    """
    if candidate.num_qubits != original.num_qubits:
        return 0.0
    if original.num_qubits > max_exact_qubits:
        return None

    o_params = _ordered_parameters(original)
    c_params = _ordered_parameters(candidate)
    # Equivalence is only meaningful when the data parameters line up.
    if len(o_params) != len(c_params):
        return 0.0

    rng = np.random.default_rng(seed)
    fids = []
    for _ in range(num_samples):
        vals = rng.uniform(0.0, np.pi, size=len(o_params))
        try:
            so = Statevector(_bind(original, vals))
            sc = Statevector(_bind(candidate, vals))
        except Exception:
            return 0.0
        fids.append(abs(so.inner(sc)) ** 2)
    return float(np.clip(np.mean(fids), 0.0, 1.0))
