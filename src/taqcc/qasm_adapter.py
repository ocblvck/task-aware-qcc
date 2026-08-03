"""Parse candidate circuits emitted by the LLM (QASM text) into Qiskit circuits.

Prefers the battle-tested helpers from the ``quantum-cirq-opt`` package
(``qcc.grpo_rewards``: ``extract_qasm_from_completion``, ``sanitize_qasm``,
``parse_qasm_circuit``) when that project is importable, since they already
handle the messy real-world output of Qwen2.5-Coder. Falls back to a minimal
local parser otherwise so the reward never hard-depends on that repo.
"""

from __future__ import annotations

import re
from typing import Optional

from qiskit import QuantumCircuit

_QCC_OK = False
try:  # pragma: no cover - availability depends on machine layout
    from qcc.grpo_rewards import (  # type: ignore
        extract_qasm_from_completion as _qcc_extract,
        sanitize_qasm as _qcc_sanitize,
        parse_qasm_circuit as _qcc_parse,
    )

    _QCC_OK = True
except Exception:
    _qcc_extract = _qcc_sanitize = _qcc_parse = None  # type: ignore


_CODE_FENCE = re.compile(r"```(?:qasm|openqasm)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_qasm(text: str) -> str:
    """Pull the QASM body out of a (possibly fenced / chatty) completion.

    Uses local extraction (code fence or the ``OPENQASM`` marker). We deliberately
    do NOT use qcc's extractor here: it is tuned for the graph-circuit project's
    QASM2-ish output and mangles parameterized OpenQASM 3.0 ``input`` circuits.
    """
    m = _CODE_FENCE.search(text)
    body = m.group(1) if m else text
    idx = body.find("OPENQASM")
    return body[idx:].strip() if idx >= 0 else body.strip()


def _local_parse(qasm: str) -> Optional[QuantumCircuit]:
    from qiskit import qasm3, qasm2

    for loader in (qasm3.loads, qasm2.loads):
        try:
            return loader(qasm)
        except Exception:
            continue
    return None


def parse_candidate(text: str) -> Optional[QuantumCircuit]:
    """Best-effort: completion text -> Qiskit circuit (or ``None`` if unparseable).

    Local OpenQASM 3.0/2.0 parsing is the PRIMARY path (our feature-map QASM is
    clean ``qasm3.dumps`` output). qcc's sanitizer is only a fallback for messy
    completions that local parsing can't handle.
    """
    qasm = extract_qasm(text)
    circ = _local_parse(qasm)
    if circ is not None:
        return circ
    if _QCC_OK:
        try:
            q = _qcc_sanitize(qasm) if _qcc_sanitize is not None else qasm
            circ = _qcc_parse(q)
            if circ is not None:
                return circ
        except Exception:
            pass
    return None

