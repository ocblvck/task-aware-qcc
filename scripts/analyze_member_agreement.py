#!/usr/bin/env python
"""How often do the three committee members get the same records wrong?

Fusing classifiers only beats the best one of them when the members are wrong about
DIFFERENT records. The conference paper assumed different feature maps would give that,
because they place the data in different parts of Hilbert space. Our results say fusion
never beat its best member, so the assumption needs checking directly.

This measures it on the noiseless kernels, where the question is still open (under noise
the answer is already known: collapsed branches all emit the majority label, so they agree
trivially). For each pair of members we report:

  * agreement       fraction of test records where both predict the same label
  * both wrong      fraction where both are wrong
  * Q statistic     Yule's Q on the correct/incorrect contingency table, +1 when two
                    members fail on exactly the same records and 0 when independent
  * disagreement    the classical diversity measure, 1 - agreement

Noiseless statevector kernels only, so this is cheap and does not touch the GPUs.

Run:
  python scripts/analyze_member_agreement.py --datasets UNSW_NB15.csv --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np

COMMITTEE = [("Z", 1, "full"), ("ZZ", 2, "full"), ("Pauli", 1, "full")]
NAMES = ["Z", "ZZ", "Pauli"]


def yule_q(a_ok, b_ok):
    """Yule's Q over the 2x2 table of who was right. +1 = identical failures."""
    n11 = float(np.sum(a_ok & b_ok))
    n00 = float(np.sum(~a_ok & ~b_ok))
    n10 = float(np.sum(a_ok & ~b_ok))
    n01 = float(np.sum(~a_ok & b_ok))
    den = n11 * n00 + n01 * n10
    return float((n11 * n00 - n01 * n10) / den) if den > 0 else 0.0


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
    ap.add_argument("--output", default="results/member_agreement.json")
    args = ap.parse_args()

    from sklearn.metrics import matthews_corrcoef
    from sklearn.svm import SVC

    from taqcc.data import load_split
    from taqcc.feature_maps import make_feature_map
    from taqcc.kernels import ideal_gram, ideal_rect

    nq = args.num_qubits
    circuits = [make_feature_map(nq, *m) for m in COMMITTEE]
    seeds = [int(s) for s in args.seeds.split(",")]
    out = {"config": vars(args), "datasets": {}}

    for ds in args.datasets.split(","):
        ds = ds.strip()
        print(f"\n[dataset] {ds}", flush=True)
        pair_stats = {f"{a}-{b}": {"agree": [], "both_wrong": [], "q": [], "disagree": []}
                      for a, b in combinations(NAMES, 2)}
        member_mcc = {n: [] for n in NAMES}
        fused_mcc, best_mcc, oracle_mcc = [], [], []

        for sd in seeds:
            X_tr, X_te, y_tr, y_te = load_split(
                str(Path(args.data_dir) / ds), nq, args.train_size, args.test_size, seed=sd)
            preds, oks = [], []
            for name, fm in zip(NAMES, circuits):
                K_tr = ideal_gram(fm, X_tr)
                K_te = ideal_rect(fm, X_te, X_tr)
                svc = SVC(kernel="precomputed", class_weight="balanced", random_state=sd)
                svc.fit(K_tr, y_tr)
                p = svc.predict(K_te)
                preds.append(p)
                oks.append(p == y_te)
                member_mcc[name].append(float(matthews_corrcoef(y_te, p)))

            for (i, a), (j, b) in combinations(list(enumerate(NAMES)), 2):
                key = f"{a}-{b}"
                same = preds[i] == preds[j]
                pair_stats[key]["agree"].append(float(np.mean(same)))
                pair_stats[key]["both_wrong"].append(float(np.mean(~oks[i] & ~oks[j])))
                pair_stats[key]["q"].append(yule_q(oks[i], oks[j]))
                pair_stats[key]["disagree"].append(float(np.mean(~same)))

            maj = (np.sum(preds, axis=0) >= 2).astype(int)
            fused_mcc.append(float(matthews_corrcoef(y_te, maj)))
            best_mcc.append(max(member_mcc[n][-1] for n in NAMES))
            # Oracle: right whenever ANY member is right. The ceiling fusion could reach
            # if the vote always picked the correct member.
            any_ok = oks[0] | oks[1] | oks[2]
            oracle = np.where(any_ok, y_te, 1 - y_te)
            oracle_mcc.append(float(matthews_corrcoef(y_te, oracle)))

        rec = {
            "member_mcc": {n: float(np.mean(v)) for n, v in member_mcc.items()},
            "majority_mcc": float(np.mean(fused_mcc)),
            "best_member_mcc": float(np.mean(best_mcc)),
            "oracle_mcc": float(np.mean(oracle_mcc)),
            "pairs": {k: {m: float(np.mean(v)) for m, v in d.items()}
                      for k, d in pair_stats.items()},
        }
        out["datasets"][ds] = rec
        print(f"  members  " + "  ".join(f"{n} {rec['member_mcc'][n]:.3f}" for n in NAMES))
        print(f"  majority {rec['majority_mcc']:.3f}   best member "
              f"{rec['best_member_mcc']:.3f}   oracle {rec['oracle_mcc']:.3f}")
        for k, d in rec["pairs"].items():
            print(f"  {k:<12} agree {d['agree']:.3f}  both wrong {d['both_wrong']:.3f}  "
                  f"Yule Q {d['q']:+.3f}")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))

    print(f"\n[done] -> {args.output}")


if __name__ == "__main__":
    main()
