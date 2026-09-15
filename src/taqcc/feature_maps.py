"""Parameterized feature maps used as *compression targets* for the LLM.

Byte-faithful copy of ``make_feature_map`` from
``quantum-ml-iot-nid/run_hardware_kernel.py`` so the new task-aware reward scores
candidates against exactly the feature maps used in the IoT-NID / UNSW study.

We intentionally COPY (rather than import) because the paper modules
(``circuit_depth_experiment.py``) do an unguarded ``from cuml.svm import SVC`` at
import time, which would drag RAPIDS/cuML into the reward path. Keeping these
helpers self-contained lets the reward run in any env that has Qiskit.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap

# Gate-level basis used for fair, decomposed metric counting. This is the SAME
# basis the paper transpiles to (run_hardware_kernel / device_noise_validation),
# so two-qubit count == CX count and metrics match the hardware ladder.
_METRIC_BASIS = ["u", "cx", "rz", "sx", "x"]

# Feature-map recipe per model, matches device_noise_validation.MODEL_MAPS and
# run_hardware_kernel.MODEL_MAPS (committee-size ablation variants included).
MODEL_MAPS: Dict[str, list] = {
    "QSVC": [("ZZ", 2)],
    "QVE": [("Z", 1), ("ZZ", 2)],
    "QWE": [("ZZ", 2), ("Pauli", 1)],
    "QVE3": [("Z", 1), ("ZZ", 2), ("Pauli", 1)],
    "QVE4": [("Z", 1), ("ZZ", 2), ("Pauli", 1), ("Custom", 1)],
}
MAJORITY_VOTE_MODELS = {"QVE", "QVE3", "QVE4"}
TWO_QUBIT_GATES = {"cx", "cz", "ecr", "swap"}


def make_feature_map(num_qubits: int, map_type: str, reps: int, entanglement: str = "full"):
    """Build a data-encoding feature map (parameter vector named ``x``)."""
    if map_type == "Z":
        return ZFeatureMap(num_qubits, reps=reps)
    if map_type == "ZZ":
        return ZZFeatureMap(num_qubits, reps=reps, entanglement=entanglement)
    if map_type == "Pauli":
        return PauliFeatureMap(
            num_qubits, reps=reps, paulis=["Z", "ZZ"], entanglement=entanglement
        )
    if map_type == "Custom":
        fm = QuantumCircuit(num_qubits)
        params = ParameterVector("x", num_qubits)
        for _ in range(reps):
            for i in range(num_qubits):
                fm.h(i)
            for i in range(num_qubits):
                fm.rz(params[i], i)
                fm.ry(params[i], i)
            if entanglement == "linear":
                for i in range(num_qubits - 1):
                    fm.cx(i, i + 1)
            else:  # full
                for i in range(num_qubits):
                    for j in range(i + 1, num_qubits):
                        fm.cx(i, j)
        return fm
    raise ValueError(f"Unknown map_type {map_type!r}")


def used_parameter_names(circuit: QuantumCircuit) -> set:
    """Names of the parameters that actually appear in a gate argument.

    ``QuantumCircuit.num_parameters`` counts *declared* parameters, including any that
    sit in the OpenQASM header and are never used. So a candidate can declare all ``n``
    data parameters, bind two of them, and still look like a full ``n``-feature encoding.
    """
    used = set()
    for inst in circuit.data:
        for prm in inst.operation.params:
            if hasattr(prm, "parameters"):
                used |= {p.name for p in prm.parameters}
    return used


def touched_qubit_indices(circuit: QuantumCircuit) -> set:
    """Indices of the qubits any instruction acts on (idle wires excluded)."""
    return {circuit.find_bit(q).index for inst in circuit.data for q in inst.qubits}


def is_valid_feature_map(circuit: Optional[QuantumCircuit], num_qubits: int) -> bool:
    """Whether a candidate really encodes all ``num_qubits`` data features.

    On top of parsing as an ``n``-qubit circuit with ``n`` declared parameters, every
    parameter has to be *used* and every qubit acted on. Without those last two checks a
    policy can max out the compression term by dropping most of the input.
    """
    if circuit is None or circuit.num_qubits != num_qubits:
        return False
    if circuit.num_parameters != num_qubits:
        return False
    if len(used_parameter_names(circuit)) != num_qubits:
        return False
    return len(touched_qubit_indices(circuit)) == num_qubits


# Perturbation sizes for the effective-parameter test (Eq. 9 in the article). Angles in
# these maps carry factors of two and products of features, so a single delta can land on
# a period of the rotation; three unrelated sizes avoid a false "inert" verdict.
EFFECTIVE_DELTAS = (0.7, 1.9, -1.1)
EFFECTIVE_TOL = 1e-9


def effective_parameter_mask(circuit: QuantumCircuit, deltas=EFFECTIVE_DELTAS,
                             tol: float = EFFECTIVE_TOL, seed: int = 0):
    """Per parameter (sorted by name): does perturbing it change the encoded state?

    The fidelity kernel is |<psi(x)|psi(z)>|^2, blind to global phase, so a parameter
    reaches the kernel only if perturbing it drops the self-fidelity below 1. A phase
    rotation on a qubit that never saw a Hadamard passes the syntactic checks and fails
    this one. Statevector only; a few dozen simulations per circuit at six qubits.
    """
    from qiskit.quantum_info import Statevector

    order = list(circuit.parameters)
    if not order:
        return []
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 2.8, len(order))

    def state(vals):
        return Statevector.from_instruction(
            circuit.assign_parameters(dict(zip(order, vals)))).data

    v0 = state(base)
    mask = []
    for prm in sorted(order, key=lambda q: q.name):
        i = order.index(prm)
        worst = 1.0
        for d in deltas:
            pert = base.copy()
            pert[i] += d
            worst = min(worst, abs(np.vdot(v0, state(pert))) ** 2)
        mask.append(bool(worst < 1 - tol))
    return mask


def is_effective_feature_map(circuit: Optional[QuantumCircuit], num_qubits: int,
                             seed: int = 0) -> bool:
    """Eq. 9: every one of the ``num_qubits`` parameters must move the encoded state.

    Stronger than :func:`is_valid_feature_map`, which it assumes has already passed.
    """
    if circuit is None:
        return False
    try:
        mask = effective_parameter_mask(circuit, seed=seed)
    except Exception:
        return False
    return len(mask) == num_qubits and all(mask)


def circuit_metrics(circuit: QuantumCircuit) -> Dict[str, int]:
    """Depth / total-gate / two-qubit-gate counts on a DECOMPOSED circuit.

    Library feature maps are single composite instructions until decomposed, so
    we transpile to a basis (opt level 0, no optimization) before counting to get
    the true gate-level cost comparable to the LLM's gate-level QASM output.
    """
    try:
        decomposed = transpile(circuit, basis_gates=_METRIC_BASIS, optimization_level=0)
    except Exception:
        decomposed = circuit.decompose(reps=3)
    ops = decomposed.count_ops()
    two_qubit = sum(c for g, c in ops.items() if g in TWO_QUBIT_GATES)
    total = sum(c for g, c in ops.items() if g not in {"barrier", "measure"})
    return {
        "depth": int(decomposed.depth()),
        "gates": int(total),
        "two_qubit": int(two_qubit),
        "num_qubits": int(decomposed.num_qubits),
    }
