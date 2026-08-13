#!/usr/bin/env python
"""Pilot: end-to-end task-aware reward (R1 + R2) on a tiny IoT/UNSW split.

Demonstrates that the two-part reward behaves as designed, on a handful of
candidate compressions of the QSVC ZZ feature map:

  * ``identity``: the original map (sanity: equiv=1, compression=0)
  * ``transpiled_l3``: Qiskit L3 of the original (should stay equivalent, smaller)
  * ``linear_zz``: ZZ with linear entanglement (fewer 2q gates, NOT equivalent)
  * ``z_only``: Z map, no entanglement (cheapest, least faithful)

Run (uses the project conda env that has qiskit-aer + sklearn):
  /home/chibuike/miniconda/envs/qiskit/bin/python scripts/pilot_reward.py \
      --dataset UNSW_NB15.csv --data-dir /home/chibuike/quantum-ml-iot-nid \
      --num-qubits 6 --train-size 16 --test-size 8 --noise-p1 0.01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make the package importable without installation, and expose qcc helpers if present.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
for _p in ("/home/chibuike/quantum-cirq-opt/src",):
    if Path(_p).exists():
        sys.path.append(_p)

import numpy as np
from qiskit import transpile

from taqcc.data import load_split
from taqcc.downstream import DownstreamConfig, downstream_accuracy
from taqcc.feature_maps import make_feature_map, circuit_metrics
from taqcc.reward import TaskAwareRewardConfig, TaskContext, score_candidate

_BASIS = ["u", "cx", "rz", "sx", "x"]


def build_candidates(num_qubits: int, entanglement: str):
    """Return {name: feature_map} candidate compressions of the QSVC ZZ map."""
    original = make_feature_map(num_qubits, "ZZ", reps=2, entanglement=entanglement)
    cands = {
        "identity": original.copy(),
        "transpiled_l3": transpile(
            original, basis_gates=_BASIS, optimization_level=3, seed_transpiler=42
        ),
        "linear_zz": make_feature_map(num_qubits, "ZZ", reps=2, entanglement="linear"),
        "z_only": make_feature_map(num_qubits, "Z", reps=2),
    }
    return original, cands


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="UNSW_NB15.csv")
    ap.add_argument("--data-dir", default="/home/chibuike/quantum-ml-iot-nid")
    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--entanglement", default="full", choices=["full", "linear"])
    ap.add_argument("--train-size", type=int, default=16)
    ap.add_argument("--test-size", type=int, default=8)
    ap.add_argument("--pool-size", type=int, default=4000)
    ap.add_argument("--noise-p1", type=float, default=0.01,
                    help="1q depolarizing strength (0 => exact statevector path)")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.perf_counter()
    path = Path(args.data_dir) / args.dataset
    print(f"[data] loading {path} -> {args.num_qubits}q "
          f"({args.train_size} train / {args.test_size} test)", flush=True)
    X_tr, X_te, y_tr, y_te = load_split(
        str(path), args.num_qubits, args.train_size, args.test_size,
        pool_size=args.pool_size, seed=args.seed,
    )
    print(f"[data] train classes={np.bincount(y_tr).tolist()} "
          f"test classes={np.bincount(y_te).tolist()}", flush=True)

    original, candidates = build_candidates(args.num_qubits, args.entanglement)
    dcfg = DownstreamConfig(
        num_qubits=args.num_qubits, noise_p1=args.noise_p1,
        seed=args.seed, gpu=not args.no_gpu,
    )
    ctx = TaskContext(
        original=original, X_train=X_tr, y_train=y_tr,
        X_test=X_te, y_test=y_te, downstream=dcfg,
    )
    rcfg = TaskAwareRewardConfig()

    base_acc = ctx.ensure_baseline()
    om = circuit_metrics(original)
    print(f"\n[baseline] original ZZ({args.entanglement}) depth={om['depth']} "
          f"2q={om['two_qubit']} | noisy(p1={args.noise_p1}) acc={base_acc:.3f}\n",
          flush=True)

    rows = []
    header = f"{'candidate':14s} {'depth':>5s} {'2q':>4s} {'comp%':>6s} " \
             f"{'equiv':>6s} {'acc':>6s} {'util':>5s} {'gate':>5s} {'reward':>7s}"
    print(header)
    print("-" * len(header))
    for name, fm in candidates.items():
        res = score_candidate(fm, ctx, rcfg)
        m = circuit_metrics(fm)
        equiv = res["equiv"]
        equiv_s = "n/a" if equiv is None else f"{equiv:.3f}"
        acc = res["candidate_accuracy"]
        acc_s = "n/a" if acc is None else f"{acc:.3f}"
        print(f"{name:14s} {m['depth']:5d} {m['two_qubit']:4d} "
              f"{100*res['compression_gain']:6.1f} {equiv_s:>6s} {acc_s:>6s} "
              f"{res['utility']:5.2f} {res['gate']:5.2f} {res['reward']:7.3f}",
              flush=True)
        rows.append({"candidate": name, "metrics": m, **res})

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset, "num_qubits": args.num_qubits,
        "entanglement": args.entanglement, "noise_p1": args.noise_p1,
        "train_size": args.train_size, "test_size": args.test_size,
        "baseline_accuracy": base_acc, "results": rows,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    out_dir = _ROOT / "pilots"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"pilot_{args.dataset.split('.')[0]}_{args.num_qubits}q_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[done] {out['elapsed_sec']}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
