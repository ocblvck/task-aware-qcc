#!/usr/bin/env python
"""Dataset and split manifests, so a third party can confirm they hold the same files and
reproduce the same train/test partitions before running anything expensive.

datasets.json  sha256, size, rows, columns and label counts of each CSV
splits.json    sha256 of the encoded X_train, X_test, y_train, y_test arrays returned by
               taqcc.data.load_split for every (dataset, width, data-split seed) the
               article uses, rounded to 10 decimals before hashing

  python scripts/make_manifests.py --data-dir /path/to/csvs
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from taqcc.data import load_split, _detect_delimiter
from taqcc.io import atomic_write_json

DS = ["IoT_Original_Distribution.csv", "UNSW_NB15.csv", "UNSW_2018_IoT_Botnet_Final_10_Best.csv"]
LABEL = {"IoT_Original_Distribution.csv": "Label", "UNSW_NB15.csv": "label", "UNSW_2018_IoT_Botnet_Final_10_Best.csv": "attack"}
PLAN = [(6, range(0, 15)), (8, range(0, 5)), (10, range(0, 5))]          # width, data-split seeds
EXTRA = [("UNSW_NB15.csv", 10, range(5, 15))]                            # fifteen-split ten-qubit cell

def sha(a): return hashlib.sha256(np.ascontiguousarray(np.round(np.asarray(a, dtype=float), 10)).tobytes()).hexdigest()[:16]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", required=True); ap.add_argument("--out", default=str(ROOT / "results/manifests"))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dsm = {}
    for f in DS:
        p = Path(a.data_dir) / f; h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""): h.update(chunk)
        df = pd.read_csv(p, sep=_detect_delimiter(str(p)), low_memory=False); df.columns = [str(c).strip() for c in df.columns]
        lab = LABEL[f] if LABEL[f] in df.columns else df.columns[-1]
        dsm[f] = {"sha256": h.hexdigest(), "bytes": p.stat().st_size, "rows": int(len(df)), "columns": int(df.shape[1]),
                  "label_column": lab, "label_counts": {str(k): int(v) for k, v in df[lab].value_counts().items()}}
        print("[dataset]", f, dsm[f]["rows"], dsm[f]["label_counts"], flush=True)
    atomic_write_json(out / "datasets.json", dsm, indent=1)
    sp = {}
    jobs = [(f, w, s) for f in DS for w, seeds in PLAN for s in seeds] + [(f, w, s) for f, w, seeds in EXTRA for s in seeds]
    for f, w, s in jobs:
        Xtr, Xte, ytr, yte = load_split(str(Path(a.data_dir) / f), w, 200, 400, seed=s)
        sp[f"{f}|{w}q|seed{s}"] = {"X_train": sha(Xtr), "X_test": sha(Xte), "y_train": sha(ytr), "y_test": sha(yte),
                                     "train_class_counts": np.bincount(ytr.astype(int)).tolist(), "test_class_counts": np.bincount(yte.astype(int)).tolist()}
        atomic_write_json(out / "splits.json", {"protocol": "load_split(path, width, 200, 400, pool_size=5000, seed)", "splits": sp}, indent=1)
        print("[split]", f, w, s, flush=True)
    rw = {}
    for s in (42, 43, 44, 45, 46):
        for tr, te in ((48, 24), (16, 8)):
            Xtr, Xte, ytr, yte = load_split(str(Path(a.data_dir) / "UNSW_NB15.csv"), 6, tr, te, pool_size=4000, seed=s)
            rw[f"reward|{tr}/{te}|seed{s}"] = {"X_train": sha(Xtr), "X_test": sha(Xte), "train_class_counts": np.bincount(ytr.astype(int)).tolist(), "test_class_counts": np.bincount(yte.astype(int)).tolist()}
    atomic_write_json(out / "reward_subsamples.json", {"protocol": "load_split(UNSW_NB15.csv, 6, train, test, pool_size=4000, seed)", "note": "every policy up to and including the revision campaign used seed 42; the supplement uses the training seed", "subsamples": rw}, indent=1)

if __name__ == "__main__":
    main()
