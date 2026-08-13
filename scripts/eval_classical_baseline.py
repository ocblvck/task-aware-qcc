#!/usr/bin/env python
"""Classical baselines under the *identical* protocol as the quantum ensembles.

The compression study runs in a small NISQ-scale regime (6 qubits, 48 training /
160 test instances, five seeds) that is deliberately different from the
conference benchmark, which makes its absolute MCC values hard to calibrate. This
script anchors that scale by running classical learners on exactly the same
angle-encoded features, splits, and seeds produced by ``taqcc.data.load_split``,
so the numbers drop straight into the ensemble table.

No noise is applied: classical models do not run on the quantum device, so their
score is the noise-free reference line across the whole depolarizing grid.

Run:
  python scripts/eval_classical_baseline.py --seeds 0,1,2,3,4 \
      --train-size 48 --test-size 160 --output results/classical_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.svm import SVC


def make_models(seed: int):
    """Classical learners matching the conference baselines, class-balanced."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=seed, class_weight="balanced"),
        "SVM_RBF": SVC(kernel="rbf", random_state=seed, class_weight="balanced"),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=seed, class_weight="balanced"),
    }


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
    ap.add_argument("--output", default="results/classical_baseline.json")
    args = ap.parse_args()

    from taqcc.data import load_split

    seeds = [int(s) for s in args.seeds.split(",")]
    out = {"config": vars(args), "datasets": {}}

    for ds in args.datasets.split(","):
        ds = ds.strip()
        print(f"\n[dataset] {ds}", flush=True)
        per_model = {name: {"acc": [], "mcc": []} for name in make_models(0)}
        for sd in seeds:
            X_tr, X_te, y_tr, y_te = load_split(
                str(Path(args.data_dir) / ds), args.num_qubits,
                args.train_size, args.test_size, seed=sd)
            for name, clf in make_models(sd).items():
                clf.fit(X_tr, y_tr)
                pred = clf.predict(X_te)
                per_model[name]["acc"].append(float(accuracy_score(y_te, pred)))
                per_model[name]["mcc"].append(float(matthews_corrcoef(y_te, pred)))
        ds_out = {}
        for name, v in per_model.items():
            ds_out[name] = {
                "acc": [float(np.mean(v["acc"])), float(np.std(v["acc"]))],
                "mcc": [float(np.mean(v["mcc"])), float(np.std(v["mcc"]))],
                "seeds": seeds,
                "acc_seeds": v["acc"],
                "mcc_seeds": v["mcc"],
            }
            print(f"  {name:20} acc {ds_out[name]['acc'][0]:.3f}"
                  f"+-{ds_out[name]['acc'][1]:.3f}  "
                  f"mcc {ds_out[name]['mcc'][0]:.3f}+-{ds_out[name]['mcc'][1]:.3f}",
                  flush=True)
        out["datasets"][ds] = ds_out

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n[done] -> {args.output}")


if __name__ == "__main__":
    main()
