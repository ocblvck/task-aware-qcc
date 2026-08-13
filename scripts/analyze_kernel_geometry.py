#!/usr/bin/env python
"""Measure the kernel geometry that the ensemble and compression claims rest on.

Two mechanisms are asserted throughout the manuscript but never measured:

  1. *Encoding-level diversity* — the Z and ZZ/Pauli branches are supposed to give
     complementary decision views. We quantify this as the pairwise disagreement
     rate between the branch predictions on real test data.
  2. *Kernel concentration* — the entangling branches are supposed to collapse
     under noise because their off-diagonal kernel values squeeze into a narrow
     band, erasing class structure. We quantify this as the spread (std, max) of
     the off-diagonal Gram entries, for the original and compressed maps, with and
     without depolarizing noise.

Run:
  python scripts/analyze_kernel_geometry.py --dataset UNSW_NB15.csv \
      --cache-dir results/compressed_maps --output results/kernel_geometry.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np

COMPONENTS = [("Z", 1, "full"), ("ZZ", 2, "full"), ("Pauli", 1, "full")]


def spread(K: np.ndarray) -> dict:
    """Off-diagonal Gram statistics: a concentrated kernel has tiny std/range."""
    off = K[~np.eye(K.shape[0], dtype=bool)]
    return {"mean": float(off.mean()), "std": float(off.std()),
            "max": float(off.max()), "min": float(off.min())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="IoT_Original_Distribution.csv,UNSW_NB15.csv,"
                                          "UNSW_2018_IoT_Botnet_Final_10_Best.csv")
    ap.add_argument("--data-dir", default="/home/chibuike/quantum-ml-iot-nid")
    ap.add_argument("--cache-dir", default="results/compressed_maps")
    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--train-size", type=int, default=48)
    ap.add_argument("--test-size", type=int, default=160)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--noise-grid", default="0.0,0.01")
    ap.add_argument("--output", default="results/kernel_geometry.json")
    args = ap.parse_args()

    from taqcc.data import load_split
    from taqcc.downstream import _fit_eval_single
    from taqcc.kernels import gram_pair
    from qiskit import qasm3
    from taqcc.qasm_adapter import parse_candidate

    def load_cached(key, kind):
        txt = (Path(args.cache_dir) / f"{key}.{kind}.qasm").read_text()
        try:
            return qasm3.loads(txt)
        except Exception:
            return parse_candidate(txt)

    circuits = {}
    for (mt, reps, ent) in COMPONENTS:
        key = f"{mt}_{reps}_{ent}"
        circuits[(mt, "orig")] = load_cached(key, "orig")
        circuits[(mt, "comp")] = load_cached(key, "comp")

    seeds = [int(s) for s in args.seeds.split(",")]
    noises = [float(x) for x in args.noise_grid.split(",")]
    out = {"config": vars(args), "datasets": {}}

    for ds in args.datasets.split(","):
        ds = ds.strip()
        print(f"\n[dataset] {ds}", flush=True)
        ds_out = {"concentration": {}, "diversity": {}}

        for p1 in noises:
            for (mt, _r, _e) in COMPONENTS:
                for kind in ("orig", "comp"):
                    stats = []
                    for sd in seeds:
                        X_tr, X_te, y_tr, y_te = load_split(
                            str(Path(args.data_dir) / ds), args.num_qubits,
                            args.train_size, args.test_size, seed=sd)
                        K_tr, _ = gram_pair(circuits[(mt, kind)], X_tr, X_te,
                                            p1=p1, gpu=False)
                        stats.append(spread(K_tr))
                    agg = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
                    ds_out["concentration"][f"{mt}_{kind}_p{p1}"] = agg
                    print(f"  p1={p1} {mt:6}/{kind:4} off-diag "
                          f"mean {agg['mean']:.4f} std {agg['std']:.4f} "
                          f"max {agg['max']:.4f}", flush=True)

        # Branch disagreement = the diversity the heterogeneous ensemble trades on.
        for p1 in noises:
            for kind in ("orig", "comp"):
                preds = {}
                for (mt, _r, _e) in COMPONENTS:
                    per_seed = []
                    for sd in seeds:
                        X_tr, X_te, y_tr, y_te = load_split(
                            str(Path(args.data_dir) / ds), args.num_qubits,
                            args.train_size, args.test_size, seed=sd)
                        K_tr, K_te = gram_pair(circuits[(mt, kind)], X_tr, X_te,
                                               p1=p1, gpu=False)
                        _, _, pred = _fit_eval_single(K_tr, K_te, y_tr, y_te, sd)
                        per_seed.append(np.asarray(pred))
                    preds[mt] = per_seed
                pairs = {}
                names = [c[0] for c in COMPONENTS]
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        a, b = names[i], names[j]
                        d = [float(np.mean(pa != pb))
                             for pa, pb in zip(preds[a], preds[b])]
                        pairs[f"{a}_vs_{b}"] = [float(np.mean(d)), float(np.std(d))]
                ds_out["diversity"][f"{kind}_p{p1}"] = pairs
                print(f"  p1={p1} {kind:4} disagreement " +
                      "  ".join(f"{k} {v[0]:.3f}" for k, v in pairs.items()),
                      flush=True)

        out["datasets"][ds] = ds_out

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n[done] -> {args.output}")


if __name__ == "__main__":
    main()
