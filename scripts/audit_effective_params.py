#!/usr/bin/env python
"""Count how many declared parameters actually move the state a fidelity kernel sees.

`is_valid_feature_map` accepts a circuit when it acts on n qubits, declares n parameters,
uses every parameter in some gate argument and touches every qubit. That is a syntactic
test on the circuit body and it does not imply a parameter has any physical effect. The
cheapest counterexample is a phase gate on a qubit that never saw a Hadamard: `rz(theta)`
on |0> is the identity up to global phase, so `theta` appears in the QASM, satisfies the
check, and contributes nothing.

That gap is exploitable. The seed-43 policy at lr 5e-6 emitted circuits scoring 92%
compression with all three members distinct and all three passing `is_valid_feature_map`,
in which four of six features never reach the kernel.

The test here is physical rather than syntactic. The fidelity kernel is
|<psi(x)|psi(z)>|^2, which is blind to global phase, so parameter i is effective exactly
when perturbing it drops the self-fidelity below 1. We try several perturbation sizes per
parameter because a single unlucky delta can land on a period of the rotation.

Statevector only, so this is seconds per circuit and does not touch the GPUs.

Run:
  python scripts/audit_effective_params.py
  python scripts/audit_effective_params.py --circuits results/replicate_circuits
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np

# Rotation angles in these maps carry factors of 2 and products of features, so a delta
# that is a multiple of pi can return the same state. Three unrelated sizes avoid that.
DELTAS = (0.7, 1.9, -1.1)
TOL = 1e-9


def effective_mask(circ, rng):
    """Boolean per parameter, sorted by name: does perturbing it change the state?"""
    from qiskit.quantum_info import Statevector

    order = list(circ.parameters)
    base = rng.uniform(0.2, 2.8, len(order))

    def state(vals):
        return Statevector.from_instruction(
            circ.assign_parameters(dict(zip(order, vals)))).data

    v0 = state(base)
    mask = []
    for p in sorted(order, key=lambda q: q.name):
        i = order.index(p)
        worst = 1.0
        for d in DELTAS:
            pert = base.copy()
            pert[i] += d
            worst = min(worst, abs(np.vdot(v0, state(pert))) ** 2)
        mask.append(bool(worst < 1 - TOL))
    return mask


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuits", default="results/replicate_circuits",
                    help="Directory of *.comp.qasm emitted by emit_replicate_circuits.py")
    ap.add_argument("--output", default="results/effective_params.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from qiskit import qasm3

    from taqcc.feature_maps import is_valid_feature_map

    rng = np.random.default_rng(args.seed)
    files = sorted(glob.glob(str(Path(args.circuits) / "*.comp.qasm")))
    if not files:
        print(f"[error] no circuits in {args.circuits}")
        return 1

    out = {}
    print(f"{'circuit':<40}{'effective':>10}{'declared':>10}{'syntactic':>11}   per-param")
    for f in files:
        name = Path(f).name.replace(".comp.qasm", "")
        circ = qasm3.loads(Path(f).read_text())
        nq = circ.num_qubits
        mask = effective_mask(circ, rng)
        syn = bool(is_valid_feature_map(circ, nq))
        out[name] = {
            "num_qubits": nq,
            "declared": len(mask),
            "effective": int(sum(mask)),
            "per_param": mask,
            "passes_syntactic_check": syn,
            "gamed": bool(syn and sum(mask) < len(mask)),
        }
        pat = "".join("y" if m else "." for m in mask)
        flag = "   <-- INERT PARAMETERS" if out[name]["gamed"] else ""
        print(f"{name:<40}{sum(mask):>10}{len(mask):>10}{str(syn):>11}   {pat}{flag}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))

    gamed = sorted(k for k, v in out.items() if v["gamed"])
    print(f"\n[written] {args.output}  ({len(out)} circuits)")
    if gamed:
        print(f"[warn] {len(gamed)} circuit(s) pass the syntactic check with inert "
              f"parameters:")
        for k in gamed:
            print(f"        {k}  {out[k]['effective']}/{out[k]['declared']}")
    else:
        print("[ok] every circuit binds every parameter to a physical effect")
    return 0


if __name__ == "__main__":
    sys.exit(main())
