#!/usr/bin/env python
"""Compare ensemble fusion rules under depolarizing noise.

The heterogeneous ensembles fail under noise not because every branch fails, but
because the *deep* branches fail together: once depolarised they all emit the
majority class, and as a bloc they outvote the shallow branch that still works.
This script measures that directly by scoring the same three encodings under three
fusion rules on the same splits:

  * ``QVE3``  — plain majority vote (the conference rule)
  * ``QWE3``  — validation-accuracy weighting (the conference adaptive rule)
  * ``NWE3``  — kernel-spread-gated weighting: drop branches whose Gram matrix has
                concentrated. Label-free, so it can run at deployment time.

Per-branch diagnostics (kernel spread, standalone MCC, retained/dropped) are
recorded so the mechanism is visible, not just the outcome.

Run:
  python scripts/eval_fusion_rules.py --seeds 0,1,2,3,4 \
      --train-size 48 --test-size 160 --output results/fusion_rules.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np

MAPS = [("Z", 1, "full"), ("ZZ", 2, "full"), ("Pauli", 1, "full")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="IoT_Original_Distribution.csv,UNSW_NB15.csv,"
                                          "UNSW_2018_IoT_Botnet_Final_10_Best.csv")
    ap.add_argument("--data-dir", default="/home/chibuike/quantum-ml-iot-nid")
    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--train-size", type=int, default=48)
    ap.add_argument("--test-size", type=int, default=160)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--noise-grid", default="0.0,0.01,0.03,0.05,0.1")
    ap.add_argument("--spread-floor", type=float, default=0.01)
    ap.add_argument("--output", default="results/fusion_rules.json")
    args = ap.parse_args()

    from taqcc.data import load_split
    from taqcc.downstream import (
        DownstreamConfig, _fit_eval_single, kernel_spread,
        downstream_accuracy_ensemble, downstream_accuracy_weighted_ensemble,
        downstream_accuracy_spread_gated_ensemble,
    )
    from taqcc.feature_maps import make_feature_map
    from taqcc.kernels import gram_pair

    circuits = [make_feature_map(args.num_qubits, mt, r, e) for mt, r, e in MAPS]
    names = [m[0] for m in MAPS]
    seeds = [int(s) for s in args.seeds.split(",")]
    noises = [float(x) for x in args.noise_grid.split(",")]
    out = {"config": vars(args), "datasets": {}}

    for ds in args.datasets.split(","):
        ds = ds.strip()
        print(f"\n[dataset] {ds}", flush=True)
        by_noise = {}
        for p1 in noises:
            acc = {k: [] for k in ("QVE3", "QWE3", "NWE3")}
            mcc = {k: [] for k in ("QVE3", "QWE3", "NWE3")}
            branch_mcc = {n: [] for n in names}
            branch_spread = {n: [] for n in names}
            kept = {n: [] for n in names}
            for sd in seeds:
                X_tr, X_te, y_tr, y_te = load_split(
                    str(Path(args.data_dir) / ds), args.num_qubits,
                    args.train_size, args.test_size, seed=sd)
                cfg = DownstreamConfig(num_qubits=args.num_qubits, noise_p1=p1,
                                       seed=sd, gpu=False)
                # Per-branch diagnostics.
                for nm, fm in zip(names, circuits):
                    Ktr, Kte = gram_pair(fm, X_tr, X_te, p1=p1, gpu=False)
                    _, m, _ = _fit_eval_single(Ktr, Kte, y_tr, y_te, sd)
                    branch_mcc[nm].append(m)
                    branch_spread[nm].append(kernel_spread(Ktr))
                r_v = downstream_accuracy_ensemble(circuits, X_tr, y_tr, X_te, y_te, cfg)
                r_w = downstream_accuracy_weighted_ensemble(circuits, X_tr, y_tr,
                                                            X_te, y_te, cfg)
                r_n = downstream_accuracy_spread_gated_ensemble(
                    circuits, X_tr, y_tr, X_te, y_te, cfg,
                    spread_floor=args.spread_floor)
                for k, r in (("QVE3", r_v), ("QWE3", r_w), ("NWE3", r_n)):
                    acc[k].append(r["accuracy"]); mcc[k].append(r["mcc"])
                for nm, k in zip(names, r_n["kept"]):
                    kept[nm].append(bool(k))

            by_noise[p1] = {
                "fusion": {k: {"acc": [float(np.mean(acc[k])), float(np.std(acc[k]))],
                               "mcc": [float(np.mean(mcc[k])), float(np.std(mcc[k]))],
                               "mcc_seeds": [float(v) for v in mcc[k]]}
                           for k in acc},
                "branch_mcc": {n: float(np.mean(v)) for n, v in branch_mcc.items()},
                "branch_spread": {n: float(np.mean(v)) for n, v in branch_spread.items()},
                "branch_kept_rate": {n: float(np.mean(v)) for n, v in kept.items()},
            }
            f = by_noise[p1]["fusion"]
            print(f"  p1={p1:<5} MCC  QVE3 {f['QVE3']['mcc'][0]:.3f}  "
                  f"QWE3 {f['QWE3']['mcc'][0]:.3f}  NWE3 {f['NWE3']['mcc'][0]:.3f}  | "
                  f"branch MCC " +
                  " ".join(f"{n} {by_noise[p1]['branch_mcc'][n]:.2f}" for n in names) +
                  " | spread " +
                  " ".join(f"{n} {by_noise[p1]['branch_spread'][n]:.3f}" for n in names),
                  flush=True)
        out["datasets"][ds] = {"by_noise": by_noise}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n[done] -> {args.output}")


if __name__ == "__main__":
    main()
