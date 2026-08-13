#!/usr/bin/env python
"""Does noise-induced kernel concentration get worse with qubit count?

The depolarised limit of a fidelity kernel is Tr[rho rho'] = 2^-n for every pair of
distinct points, so the theory predicts that a branch which has concentrated sits at
a floor that falls *exponentially* with the number of qubits, and that wider circuits
reach that floor at lower noise. If so, the collapse we measure at six qubits is not
a small-scale artefact -- it is milder than what the conference study's 10--16 qubit
configurations would face.

For each qubit count we record, per branch and noise level, the spread of the
training Gram matrix (label-free concentration diagnostic), its mean (which should
approach 2^-n), and the branch's standalone MCC.

Run (one qubit count per process, parallelised by the caller):
  python scripts/eval_scale_concentration.py --num-qubits 8 --seeds 0,1,2 \
      --noise-grid 0.0,0.01,0.03 --test-size 64 --output results/scale_8q.json
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
    ap.add_argument("--dataset", default="UNSW_NB15.csv")
    ap.add_argument("--data-dir", default="/home/chibuike/quantum-ml-iot-nid")
    ap.add_argument("--num-qubits", type=int, required=True)
    ap.add_argument("--train-size", type=int, default=48)
    ap.add_argument("--test-size", type=int, default=64)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--noise-grid", default="0.0,0.01,0.03",
                    help="p1 values; p2 defaults to 10*p1 inside the noise model")
    ap.add_argument("--noise-pairs", default=None,
                    help="Explicit 'p1:p2,...' pairs, overriding --noise-grid. Needed to "
                         "test hardware-realistic settings: the default p2=10*p1 coupling "
                         "makes p1=0.01 mean a 10%% two-qubit error rate, an order of "
                         "magnitude worse than current superconducting devices.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from taqcc.data import load_split
    from taqcc.downstream import _fit_eval_single, kernel_spread
    from taqcc.feature_maps import make_feature_map, circuit_metrics
    from taqcc.kernels import gram_pair

    nq = args.num_qubits
    circuits = {m[0]: make_feature_map(nq, *m) for m in MAPS}
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.noise_pairs:
        noises = [tuple(float(v) for v in pair.split(":"))
                  for pair in args.noise_pairs.split(",")]
    else:
        noises = [(float(x), None) for x in args.noise_grid.split(",")]
    floor = 2.0 ** (-nq)

    out = {"config": vars(args), "depolarised_floor": floor,
           "circuit_metrics": {n: circuit_metrics(c) for n, c in circuits.items()},
           "by_noise": {}}
    print(f"[{nq}q] depolarised-limit kernel value 2^-{nq} = {floor:.6f}", flush=True)

    for p1, p2 in noises:
        rec = {}
        eff_p2 = (10.0 * p1) if p2 is None else p2
        for name, fm in circuits.items():
            spreads, means, mccs = [], [], []
            for sd in seeds:
                X_tr, X_te, y_tr, y_te = load_split(
                    str(Path(args.data_dir) / args.dataset), nq,
                    args.train_size, args.test_size, seed=sd)
                K_tr, K_te = gram_pair(fm, X_tr, X_te, p1=p1, p2=p2, gpu=False)
                off = K_tr[~np.eye(K_tr.shape[0], dtype=bool)]
                spreads.append(kernel_spread(K_tr))
                means.append(float(off.mean()))
                _, m, _ = _fit_eval_single(K_tr, K_te, y_tr, y_te, sd)
                mccs.append(m)
            rec[name] = {"spread": float(np.mean(spreads)),
                         "mean_offdiag": float(np.mean(means)),
                         "mcc": float(np.mean(mccs)),
                         "mcc_seeds": [float(v) for v in mccs]}
            print(f"[{nq}q] p1={p1:<7} p2={eff_p2:<6} {name:6} "
                  f"spread {rec[name]['spread']:.5f}  "
                  f"mean {rec[name]['mean_offdiag']:.5f} (floor {floor:.5f})  "
                  f"MCC {rec[name]['mcc']:.3f}", flush=True)
        rec["_two_qubit_error"] = eff_p2
        out["by_noise"][f"{p1}:{eff_p2}"] = rec
        # Flush after every noise setting: this box reboots unpredictably and a
        # write-at-the-end script loses the whole run.
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"[{nq}q] [done] -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
