#!/usr/bin/env python
"""Compression arms scored on the same protocol as the fusion study.

The old compression runs used 48 train / 160 test while the fusion study used
200 / 400, so the two sets of numbers were never comparable. This scores every
compression arm at the fusion protocol, on the circuits cached after the validity
fix, under all three fusion rules (QVE3, QWE3, NWE3).

Arms:
  uncompressed        the library maps (reference)
  l3                  Qiskit optimisation level 3 (a no-op on these maps)
  linear              hand-designed full -> linear entanglement substitution
  zonly               entanglement dropped altogether (every member -> Z)
  sft, lr5, lr75, lr10  the learned policies, post-validity-fix circuits

Each distinct circuit is simulated once per (dataset, noise, seed) and shared by
every arm that uses it. The arms overlap heavily, so this is what keeps the full
cross-product cheap enough to run.

Run:
  python scripts/eval_compression_matched.py --datasets UNSW_NB15.csv \
      --train-size 200 --test-size 400 --seeds 0,1,2,3,4 \
      --output results/compression_matched_UNSW_NB15.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np

# The conference QVE3/QWE3 committee, named as they appear in the QASM caches.
COMMITTEE = [("Z", 1, "full"), ("ZZ", 2, "full"), ("Pauli", 1, "full")]
CACHE_TAGS = ["Z_1_full", "ZZ_2_full", "Pauli_1_full"]
LEARNED = ["sft", "lr5", "lr75", "lr10"]
# Directories holding post-validity-fix circuits, searched in order.
CACHE_DIRS = ["results/singlemap_fix_UNSW_NB1", "results/singlemap_fix_UNSW_201"]


def fuse(preds, weights):
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    return ((w[:, None] * np.asarray(preds)).sum(0) >= 0.5).astype(int)


def circuit_key(circ) -> str:
    """Stable identity for a parameterised circuit, used to share Gram matrices."""
    from qiskit import qasm3
    return hashlib.md5(qasm3.dumps(circ).encode()).hexdigest()


def build_arms(nq):
    """Return {arm: [circuit per committee member]} plus per-arm two-qubit counts."""
    from qiskit import qasm3, transpile
    from taqcc.feature_maps import make_feature_map, circuit_metrics, is_valid_feature_map

    arms = {}
    arms["uncompressed"] = [make_feature_map(nq, *m) for m in COMMITTEE]

    # Qiskit level 3 on the decomposed original.
    l3 = []
    for m in COMMITTEE:
        fm = make_feature_map(nq, *m).decompose()
        l3.append(transpile(fm, basis_gates=["rz", "sx", "x", "cx"], optimization_level=3))
    arms["l3"] = l3

    # Hand-designed substitutions.
    arms["linear"] = [make_feature_map(nq, m[0], m[1], "linear") for m in COMMITTEE]
    arms["zonly"] = [make_feature_map(nq, "Z", 1, "full") for _ in COMMITTEE]

    for model in LEARNED:
        circs, ok = [], True
        for tag in CACHE_TAGS:
            path = next((Path(d) / f"{model}__{tag}.comp.qasm" for d in CACHE_DIRS
                         if (Path(d) / f"{model}__{tag}.comp.qasm").exists()), None)
            if path is None:
                ok = False
                break
            circs.append(qasm3.loads(path.read_text()))
        if not ok:
            print(f"[skip] {model}: no cached circuits", flush=True)
            continue
        bad = [c for c in circs if not is_valid_feature_map(c, nq)]
        if bad:
            print(f"[skip] {model}: {len(bad)} circuit(s) fail the validity criterion",
                  flush=True)
            continue
        arms[model] = circs

    meta = {}
    for arm, circs in arms.items():
        two_q = [circuit_metrics(c)["two_qubit"] for c in circs]
        depth = [circuit_metrics(c)["depth"] for c in circs]
        base = [circuit_metrics(c)["two_qubit"] for c in arms["uncompressed"]]
        tot_o, tot_c = sum(base), sum(two_q)
        meta[arm] = {
            "two_qubit": two_q, "depth": depth,
            "two_qubit_total": tot_c,
            "reduction_pct": 0.0 if tot_o == 0 else 100.0 * (tot_o - tot_c) / tot_o,
            "distinct_members": len({circuit_key(c) for c in circs}),
        }
    return arms, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="UNSW_NB15.csv")
    ap.add_argument("--data-dir", default="/home/chibuike/quantum-ml-iot-nid")
    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--train-size", type=int, default=200)
    ap.add_argument("--test-size", type=int, default=400)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--noise-grid", default="0.0,0.01,0.03")
    ap.add_argument("--noise-pairs", default=None)
    ap.add_argument("--primary-frac", type=float, default=0.05)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from sklearn.metrics import accuracy_score, matthews_corrcoef
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    from taqcc.data import load_split
    from taqcc.downstream import kernel_spread
    from taqcc.kernels import gram_pair

    nq = args.num_qubits
    arms, meta = build_arms(nq)
    print(f"[arms] {len(arms)}: " + ", ".join(
        f"{a}({meta[a]['two_qubit_total']}x2q, {meta[a]['distinct_members']} distinct)"
        for a in arms), flush=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    if args.noise_pairs:
        noises = [tuple(float(v) for v in p.split(":")) for p in args.noise_pairs.split(",")]
    else:
        noises = [(float(x), None) for x in args.noise_grid.split(",")]

    def fit_predict(K_tr, K_te, y_tr, seed):
        svc = SVC(kernel="precomputed", class_weight="balanced", random_state=seed)
        svc.fit(K_tr, y_tr)
        return svc.predict(K_te)

    out = {"config": vars(args), "arm_metrics": meta, "datasets": {}}

    for ds in args.datasets.split(","):
        ds = ds.strip()
        print(f"\n[dataset] {ds}", flush=True)
        ds_rec = {"by_noise": {}}
        ref_spread = {}  # (arm_member_key, seed) -> noise-free spread

        for p1, p2 in noises:
            key_p = f"{p1}:{p2}" if p2 is not None else f"{p1}"
            noiseless = (p1 == 0.0 and (p2 in (0.0, None)))
            acc_rec = {a: {k: [] for k in ("QVE3", "QWE3", "NWE3")} for a in arms}
            branch_rec = {a: [[] for _ in COMMITTEE] for a in arms}

            for sd in seeds:
                X_tr, X_te, y_tr, y_te = load_split(
                    str(Path(args.data_dir) / ds), nq,
                    args.train_size, args.test_size, seed=sd)
                # Shared Gram cache: one simulation per DISTINCT circuit.
                gram = {}
                for circs in arms.values():
                    for c in circs:
                        k = circuit_key(c)
                        if k not in gram:
                            gram[k] = gram_pair(c, X_tr, X_te, p1=p1, p2=p2,
                                                gpu=not args.no_gpu)
                strat = y_tr if len(np.unique(y_tr)) > 1 else None
                itr, ival = train_test_split(np.arange(len(y_tr)), test_size=0.2,
                                             random_state=sd, stratify=strat)

                for arm, circs in arms.items():
                    keys = [circuit_key(c) for c in circs]
                    preds, spreads, w_acc = [], [], []
                    for i, k in enumerate(keys):
                        K_tr, K_te = gram[k]
                        p = fit_predict(K_tr, K_te, y_tr, sd)
                        preds.append(p)
                        branch_rec[arm][i].append(
                            matthews_corrcoef(y_te, p) if len(np.unique(y_te)) > 1 else 0.0)
                        s = kernel_spread(K_tr)
                        spreads.append(s)
                        if noiseless:
                            ref_spread[(arm, i, sd)] = s
                        pv = fit_predict(K_tr[np.ix_(itr, itr)], K_tr[np.ix_(ival, itr)],
                                         y_tr[itr], sd)
                        w_acc.append(max(accuracy_score(y_tr[ival], pv), 1e-6))

                    acc_rec[arm]["QVE3"].append(
                        matthews_corrcoef(y_te, fuse(preds, np.ones(len(preds)))))
                    acc_rec[arm]["QWE3"].append(matthews_corrcoef(y_te, fuse(preds, w_acc)))
                    w = [1.0 if s >= args.primary_frac * ref_spread.get((arm, i, sd), s)
                         else 0.0 for i, s in enumerate(spreads)]
                    if not any(w):
                        w = [1.0 if i == int(np.argmax(spreads)) else 0.0
                             for i in range(len(spreads))]
                    acc_rec[arm]["NWE3"].append(matthews_corrcoef(y_te, fuse(preds, w)))

            ds_rec["by_noise"][key_p] = {
                a: {"fusion": {r: {"mcc_mean": float(np.mean(v)),
                                   "mcc_std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                                   "mcc_seeds": [float(x) for x in v]}
                               for r, v in acc_rec[a].items()},
                    "branch_mcc": [float(np.mean(b)) for b in branch_rec[a]]}
                for a in arms}
            out["datasets"][ds] = ds_rec
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(out, indent=2))

            print(f"  {key_p}", flush=True)
            for a in arms:
                f0 = ds_rec["by_noise"][key_p][a]["fusion"]
                print(f"    {a:<14} 2q={meta[a]['two_qubit_total']:>3} "
                      f"({meta[a]['reduction_pct']:5.1f}%) distinct={meta[a]['distinct_members']} "
                      f"| QVE3 {f0['QVE3']['mcc_mean']:.3f}  QWE3 {f0['QWE3']['mcc_mean']:.3f}  "
                      f"NWE3 {f0['NWE3']['mcc_mean']:.3f}  | branch " +
                      " ".join(f"{v:.2f}" for v in ds_rec["by_noise"][key_p][a]["branch_mcc"]),
                      flush=True)

    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n[done] -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
