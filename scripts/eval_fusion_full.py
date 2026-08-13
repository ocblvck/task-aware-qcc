#!/usr/bin/env python
"""Definitive fusion-rule study: larger samples, derived threshold, paired tests.

Supersedes ``eval_fusion_rules.py``. Three changes make it submission-grade:

  1. *Sample size.* Each branch's train/test Gram is computed ONCE per
     (dataset, noise, seed) and shared by every fusion rule; QWE's internal
     validation Grams are slices of the training Gram rather than fresh
     simulations. That removes ~5x of redundant density-matrix work, which is what
     forced the 48-sample splits, so the study can run at realistic sizes.
  2. *Derived threshold.* A fully depolarised kernel sits at Tr[rho rho'] = 2^-n for
     every distinct pair, so its off-diagonal spread is exactly zero. Rather than
     hard-coding a cutoff we express it as a fraction of the branch's own
     noise-free spread, and sweep that fraction to show the choice is not load-bearing.
  3. *Encoding ablation.* Branch diagnostics cover a depth-matched, entanglement-free
     encoding (Z with reps=2) alongside the entangling maps, which separates "shallow
     helps" from "no entanglement helps" -- the confound the conference paper flagged.

Paired Wilcoxon tests across shared seeds compare the fusion rules.

Run:
  python scripts/eval_fusion_full.py --train-size 200 --test-size 400 \
      --seeds 0,1,2,3,4 --output results/fusion_full.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np

# Committee used by the fusion rules (the conference QVE3/QWE3 membership).
COMMITTEE = [("Z", 1, "full"), ("ZZ", 2, "full"), ("Pauli", 1, "full")]
# Extra encodings scored per-branch only, to separate depth from entanglement.
# Z reps=12 (depth 24) and reps=24 (depth 48) are depth-matched to ZZ-linear
# (depth 25) and ZZ-full (depth 49) respectively, but carry NO two-qubit gates, so
# comparing the pairs isolates entanglement from circuit depth.
ABLATION = [("ZZ", 2, "linear"), ("Z", 12, "full"), ("Z", 24, "full")]


def fuse(preds, weights):
    """Weighted binary vote; weights need not be normalised."""
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    return ((w[:, None] * np.asarray(preds)).sum(0) >= 0.5).astype(int)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="IoT_Original_Distribution.csv,UNSW_NB15.csv,"
                                          "UNSW_2018_IoT_Botnet_Final_10_Best.csv")
    ap.add_argument("--data-dir", default="/home/chibuike/quantum-ml-iot-nid")
    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--train-size", type=int, default=200)
    ap.add_argument("--test-size", type=int, default=400)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--noise-grid", default="0.0,0.01,0.03,0.05,0.1")
    ap.add_argument("--noise-pairs", default=None,
                    help="Explicit 'p1:p2,...' pairs overriding --noise-grid. The default "
                         "p2=10*p1 coupling makes p1=0.01 a 10%% two-qubit error rate; real "
                         "devices are nearer 0.5-2%%, which only breaks wide circuits.")
    ap.add_argument("--spread-fracs", default="0.01,0.02,0.05,0.10,0.20",
                    help="NWE cutoffs as a fraction of each branch's noise-free spread")
    ap.add_argument("--primary-frac", type=float, default=0.05)
    ap.add_argument("--no-ablation", action="store_true",
                    help="Score only the committee (skips the depth-matched encodings)")
    ap.add_argument("--no-gpu", action="store_true",
                    help="Force the density-matrix simulator onto CPU Aer")
    ap.add_argument("--output", default="results/fusion_full.json")
    args = ap.parse_args()

    from sklearn.metrics import accuracy_score, matthews_corrcoef
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from scipy.stats import wilcoxon

    from taqcc.data import load_split
    from taqcc.downstream import kernel_spread
    from taqcc.feature_maps import make_feature_map
    from taqcc.kernels import gram_pair

    nq = args.num_qubits
    all_maps = COMMITTEE if args.no_ablation else COMMITTEE + ABLATION
    labels = [f"{m[0]}{m[1]}{'L' if m[2]=='linear' else ''}" for m in all_maps]
    circuits = {lab: make_feature_map(nq, *m) for lab, m in zip(labels, all_maps)}
    comm = labels[:len(COMMITTEE)]
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.noise_pairs:
        noises = [tuple(float(v) for v in pr.split(":")) for pr in args.noise_pairs.split(",")]
    else:
        noises = [(float(x), None) for x in args.noise_grid.split(",")]
    fracs = [float(x) for x in args.spread_fracs.split(",")]

    def fit_predict(K_tr, K_te, y_tr, seed):
        svc = SVC(kernel="precomputed", class_weight="balanced", random_state=seed)
        svc.fit(K_tr, y_tr)
        return svc.predict(K_te)

    out = {"config": vars(args), "floor_2_pow_-n": 2.0 ** (-nq), "datasets": {}}

    for ds in args.datasets.split(","):
        ds = ds.strip()
        print(f"\n[dataset] {ds}", flush=True)
        # Noise-free spread per branch per seed — the reference the cutoff scales.
        ref_spread = {lab: {} for lab in labels}
        ds_rec = {"by_noise": {}, "tests": {}}

        for p1, p2 in noises:
            key_p = f"{p1}:{p2}" if p2 is not None else f"{p1}"
            noiseless = (p1 == 0.0 and (p2 in (0.0, None)))
            per_seed = {k: [] for k in ("QVE3", "QWE3")}
            per_seed.update({f"NWE3@{f}": [] for f in fracs})
            branch = {lab: {"mcc": [], "spread": []} for lab in labels}
            kept_rate = {lab: [] for lab in comm}

            for sd in seeds:
                X_tr, X_te, y_tr, y_te = load_split(
                    str(Path(args.data_dir) / ds), nq,
                    args.train_size, args.test_size, seed=sd)
                # One Gram per branch, reused by every fusion rule below.
                K = {}
                for lab in labels:
                    K[lab] = gram_pair(circuits[lab], X_tr, X_te, p1=p1, p2=p2,
                                       gpu=not args.no_gpu)
                    s = kernel_spread(K[lab][0])
                    branch[lab]["spread"].append(s)
                    if noiseless:
                        ref_spread[lab][sd] = s
                    p = fit_predict(K[lab][0], K[lab][1], y_tr, sd)
                    branch[lab]["mcc"].append(
                        matthews_corrcoef(y_te, p) if len(np.unique(y_te)) > 1 else 0.0)

                preds = [fit_predict(K[l][0], K[l][1], y_tr, sd) for l in comm]

                # QVE3 — plain majority.
                per_seed["QVE3"].append(matthews_corrcoef(y_te, fuse(preds, np.ones(len(comm)))))

                # QWE3 — validation-accuracy weights, from SLICES of the train Gram.
                strat = y_tr if len(np.unique(y_tr)) > 1 else None
                idx = np.arange(len(y_tr))
                itr, ival = train_test_split(idx, test_size=0.2, random_state=sd,
                                             stratify=strat)
                w_acc = []
                for lab in comm:
                    Ktr = K[lab][0]
                    pv = fit_predict(Ktr[np.ix_(itr, itr)], Ktr[np.ix_(ival, itr)],
                                     y_tr[itr], sd)
                    w_acc.append(max(accuracy_score(y_tr[ival], pv), 1e-6))
                per_seed["QWE3"].append(matthews_corrcoef(y_te, fuse(preds, w_acc)))

                # NWE3 — drop branches whose spread fell below frac * its own p=0 spread.
                # Binary gate, then EQUAL votes among survivors. Weighting in
                # proportion to spread was measurably worse: spread decays faster
                # than accuracy, so a degraded-but-best branch gets under-weighted.
                for f in fracs:
                    w = []
                    for lab in comm:
                        s = kernel_spread(K[lab][0])
                        thr = f * ref_spread[lab].get(sd, s)
                        w.append(1.0 if s >= thr else 0.0)
                    if not any(w):
                        w = [1.0 if i == int(np.argmax([kernel_spread(K[l][0])
                                                        for l in comm])) else 0.0
                             for i in range(len(comm))]
                    per_seed[f"NWE3@{f}"].append(matthews_corrcoef(y_te, fuse(preds, w)))
                    if f == args.primary_frac:
                        for lab, wi in zip(comm, w):
                            kept_rate[lab].append(float(wi > 0))

            ds_rec["by_noise"][key_p] = {
                "fusion": {k: {"mcc_mean": float(np.mean(v)), "mcc_std": float(np.std(v)),
                               "mcc_seeds": [float(x) for x in v]}
                           for k, v in per_seed.items()},
                "branch": {lab: {"mcc": float(np.mean(b["mcc"])),
                                 "spread": float(np.mean(b["spread"]))}
                           for lab, b in branch.items()},
                "kept_rate": {lab: float(np.mean(v)) for lab, v in kept_rate.items()},
            }
            # Flush after every noise level: an abrupt power loss then costs one
            # cell rather than the whole run.
            out["datasets"][ds] = ds_rec
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(out, indent=2))

            f0 = ds_rec["by_noise"][key_p]["fusion"]
            print(f"  {key_p:<14} QVE3 {f0['QVE3']['mcc_mean']:.3f}  "
                  f"QWE3 {f0['QWE3']['mcc_mean']:.3f}  "
                  f"NWE3 {f0[f'NWE3@{args.primary_frac}']['mcc_mean']:.3f}  | branch " +
                  " ".join(f"{l} {ds_rec['by_noise'][key_p]['branch'][l]['mcc']:.2f}"
                           for l in labels), flush=True)

        # Paired Wilcoxon across all (noise, seed) pairs: NWE vs each alternative.
        key = f"NWE3@{args.primary_frac}"
        for other in ("QVE3", "QWE3"):
            a, b = [], []
            for kp in ds_rec["by_noise"]:
                a += ds_rec["by_noise"][kp]["fusion"][key]["mcc_seeds"]
                b += ds_rec["by_noise"][kp]["fusion"][other]["mcc_seeds"]
            diff = np.asarray(a) - np.asarray(b)
            if np.allclose(diff, 0):
                ds_rec["tests"][f"{key}_vs_{other}"] = {
                    "median_delta": 0.0, "p_value": 1.0, "n": len(diff),
                    "note": "identical across all pairs"}
            else:
                st, p = wilcoxon(a, b, zero_method="zsplit")
                ds_rec["tests"][f"{key}_vs_{other}"] = {
                    "median_delta": float(np.median(diff)), "statistic": float(st),
                    "p_value": float(p), "n": int(len(diff))}
            t = ds_rec["tests"][f"{key}_vs_{other}"]
            print(f"  [test] {key} vs {other}: median dMCC {t['median_delta']:+.3f} "
                  f"p={t['p_value']:.4g} (n={t['n']})", flush=True)

        out["datasets"][ds] = ds_rec

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n[done] -> {args.output}")


if __name__ == "__main__":
    main()
