#!/usr/bin/env python
"""Emit and audit the committee circuits from a set of trained policies.

The lr sweep was one run per setting, so we cannot tell a real compression/diversity
trade-off from training noise. Checking that needs the emitted circuits rather than the
adapters. This generates the committee for each model, checks every circuit against
``is_valid_feature_map``, and writes one summary JSON with the two numbers the paper
quotes: total 2q count and how many members are still distinct circuits.

Run:
  python scripts/emit_replicate_circuits.py \
      --models models/rep_lr5_s43 models/rep_lr5_s44 \
      --output results/replicates_structure.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

# The conference QVE3/QWE3 committee.
COMMITTEE = [("Z", 1, "full"), ("ZZ", 2, "full"), ("Pauli", 1, "full")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="Adapter directories, one per trained policy")
    ap.add_argument("--base", default="models/sft_compress_e2_merged")
    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--cache-dir", default="results/replicate_circuits")
    ap.add_argument("--max-new-tokens", type=int, default=2600)
    ap.add_argument("--output", default="results/replicates_structure.json")
    args = ap.parse_args()

    sys.path.insert(0, str(_ROOT / "scripts"))
    from eval_robust_singlemap import cache_model_maps
    from taqcc.feature_maps import make_feature_map, circuit_metrics, is_valid_feature_map
    from qiskit import qasm3

    out_path = Path(args.output)
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    base_2q = sum(circuit_metrics(make_feature_map(args.num_qubits, *m))["two_qubit"]
                  for m in COMMITTEE)

    for mdir in args.models:
        label = Path(mdir).name
        if label in out:
            print(f"[skip] {label} already summarised", flush=True)
            continue
        if not Path(mdir).exists():
            print(f"[skip] {label}: adapter directory missing", flush=True)
            continue
        print(f"\n[emit] {label}", flush=True)
        try:
            cache_model_maps(args.base, label, mdir, COMMITTEE,
                             args.num_qubits, args.cache_dir,
                             max_new_tokens=args.max_new_tokens)
        except Exception as exc:                      # keep the queue alive
            print(f"[fail] {label}: {exc}", flush=True)
            out[label] = {"error": str(exc)}
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out, indent=2))
            continue

        per_member, hashes, valid = [], [], []
        for (mt, reps, ent) in COMMITTEE:
            f = Path(args.cache_dir) / f"{label}__{mt}_{reps}_{ent}.comp.qasm"
            text = f.read_text()
            circ = qasm3.loads(text)
            per_member.append(circuit_metrics(circ)["two_qubit"])
            hashes.append(hashlib.md5(text.encode()).hexdigest())
            valid.append(bool(is_valid_feature_map(circ, args.num_qubits)))
        total = sum(per_member)
        out[label] = {
            "per_member_2q": per_member,
            "total_2q": total,
            "reduction_pct": 0.0 if base_2q == 0 else 100.0 * (base_2q - total) / base_2q,
            "distinct_members": len(set(hashes)),
            "all_valid": all(valid),
            "member_md5": [h[:8] for h in hashes],
        }
        print(f"[done] {label}: 2q {per_member} total {total} "
              f"({out[label]['reduction_pct']:.1f}%) distinct "
              f"{out[label]['distinct_members']}/3 valid={all(valid)}", flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))

    print(f"\n[written] {out_path}  ({len(out)} models)")
    print(f"{'model':<22}{'total 2q':>9}{'reduction':>11}{'distinct':>10}{'valid':>7}")
    for k, v in sorted(out.items()):
        if "error" in v:
            print(f"{k:<22}{'ERROR':>9}")
            continue
        print(f"{k:<22}{v['total_2q']:>9}{v['reduction_pct']:>10.1f}%"
              f"{v['distinct_members']:>10}{str(v['all_valid']):>7}")


if __name__ == "__main__":
    main()
