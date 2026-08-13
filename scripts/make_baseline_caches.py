#!/usr/bin/env python
"""Build non-learned compression baselines as drop-in circuit caches.

``eval_robust_ensemble.py`` loads each component map from ``{key}.comp.qasm`` in
its ``--cache-dir`` and skips generation whenever ``compressed_meta.json`` already
exists. Writing baseline circuits into their own cache directories therefore lets
the *identical* evaluation harness score a non-learned compressor, with no model
inference involved.

Three arms, answering the three obvious "why do you need an LLM?" objections:

  * ``l3``     — Qiskit transpilation at optimization level 3 (the classical
                 optimizing-compiler baseline; provably equivalent).
  * ``linear`` — the hand-designed linear-entanglement variant of each
                 full-entanglement map (~half the two-qubit gates, NOT equivalent).
                 This is what the SFT warm-up is trained to imitate.
  * ``zonly``  — every entangling map replaced by the entanglement-free Z map of
                 the same depth, i.e. a homogeneous Z ensemble. The strongest
                 control: it asks whether the entangling branches were needed at all.

Run:
  python scripts/make_baseline_caches.py --num-qubits 6 --out-root results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from qiskit import transpile

from taqcc.feature_maps import make_feature_map, circuit_metrics, _METRIC_BASIS
from taqcc.grpo_integration import feature_map_to_qasm

# Same component maps the ensembles in eval_robust_ensemble.py are built from.
COMPONENTS = [("Z", 1, "full"), ("ZZ", 2, "full"), ("Pauli", 1, "full"),
              ("Custom", 1, "full"), ("ZZ", 2, "linear")]


def baseline_circuit(arm: str, num_qubits: int, map_type: str, reps: int, ent: str):
    """Return the baseline replacement circuit for one component map."""
    fm = make_feature_map(num_qubits, map_type, reps, ent)
    if arm == "l3":
        return transpile(fm, basis_gates=_METRIC_BASIS, optimization_level=3,
                         seed_transpiler=42)
    if arm == "linear":
        # Only full-entanglement sources have a cheaper linear variant; the Z map
        # has no entanglement and the linear ZZ is already minimal.
        if ent == "full" and map_type != "Z":
            return make_feature_map(num_qubits, map_type, reps, "linear")
        return fm
    if arm == "zonly":
        # Strip entanglement entirely, keeping the encoding depth (reps).
        return make_feature_map(num_qubits, "Z", reps)
    raise ValueError(f"unknown arm {arm!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--arms", default="l3,linear,zonly")
    args = ap.parse_args()

    for arm in args.arms.split(","):
        arm = arm.strip()
        cache_dir = Path(args.out_root) / f"baseline_{arm}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta = {}
        for (mt, reps, ent) in COMPONENTS:
            fm = make_feature_map(args.num_qubits, mt, reps, ent)
            bl = baseline_circuit(arm, args.num_qubits, mt, reps, ent)
            key = f"{mt}_{reps}_{ent}"
            (cache_dir / f"{key}.orig.qasm").write_text(feature_map_to_qasm(fm))
            (cache_dir / f"{key}.comp.qasm").write_text(feature_map_to_qasm(bl))
            meta[key] = {"orig_2q": circuit_metrics(fm)["two_qubit"],
                         "comp_2q": circuit_metrics(bl)["two_qubit"],
                         "compressed_ok": True}
            m = meta[key]
            print(f"[{arm}] {key}: 2q {m['orig_2q']}->{m['comp_2q']} "
                  f"depth {circuit_metrics(fm)['depth']}->{circuit_metrics(bl)['depth']}",
                  flush=True)
        (cache_dir / "compressed_meta.json").write_text(json.dumps(meta, indent=2))
        tot_o = sum(v["orig_2q"] for v in meta.values())
        tot_c = sum(v["comp_2q"] for v in meta.values())
        pct = 100 * (tot_o - tot_c) / tot_o if tot_o else 0.0
        print(f"[{arm}] total two-qubit {tot_o}->{tot_c} = {pct:.1f}% reduction "
              f"-> {cache_dir}\n", flush=True)


if __name__ == "__main__":
    main()
