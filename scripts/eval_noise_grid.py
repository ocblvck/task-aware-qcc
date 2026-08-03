#!/usr/bin/env python
"""Noise-robustness sweep for compressed feature maps (paper rigor check).

For each model and feature-map spec we generate ONE compressed circuit (greedy),
then evaluate its downstream QSVC accuracy + retention-vs-original across a grid
of depolarizing noise levels. Compression is noise-independent (a property of the
circuit), so we generate once and re-score cheaply per noise level.

Run:
  python scripts/eval_noise_grid.py --base models/sft_compress_e2_merged \
     --model sft: --model lr5:models/grpo_e2_lr5 \
     --dataset UNSW_NB15.csv --num-qubits 6 --train-size 32 --test-size 32 \
     --noise-grid 0.0,0.01,0.03,0.05,0.1 --output results/noise_grid_lr5_unsw.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
for _p in ("/home/chibuike/quantum-cirq-opt/src",):
    if Path(_p).exists() and _p not in sys.path:
        sys.path.append(_p)

import numpy as np


def dedup_specs():
    from taqcc.feature_maps import MODEL_MAPS
    seen, specs = set(), []
    for maps in MODEL_MAPS.values():
        for mt, reps in maps:
            for ent in ("full", "linear"):
                if (mt, reps, ent) not in seen:
                    seen.add((mt, reps, ent))
                    specs.append((mt, reps, ent))
    return specs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", action="append", default=[], help="label:adapter (empty=base)")
    ap.add_argument("--dataset", default="UNSW_NB15.csv")
    ap.add_argument("--data-dir", default="/home/chibuike/quantum-ml-iot-nid")
    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--train-size", type=int, default=32)
    ap.add_argument("--test-size", type=int, default=32)
    ap.add_argument("--noise-grid", default="0.0,0.01,0.03,0.05,0.1")
    ap.add_argument("--gate-mode", default="or", choices=["and", "or"])
    ap.add_argument("--max-new-tokens", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/noise_grid.json")
    args = ap.parse_args()

    import torch
    import transformers
    from peft import PeftModel

    from taqcc.data import load_split
    from taqcc.downstream import DownstreamConfig
    from taqcc.feature_maps import make_feature_map, circuit_metrics
    from taqcc.grpo_integration import (
        _compression_prompt, _TASK_SYSTEM_PROMPT, feature_map_to_qasm,
    )
    from taqcc.reward import TaskAwareRewardConfig, TaskContext, score_candidate

    noises = [float(x) for x in args.noise_grid.split(",")]
    specs = dedup_specs()
    dpath = str(Path(args.data_dir) / args.dataset)
    X_tr, X_te, y_tr, y_te = load_split(
        dpath, args.num_qubits, args.train_size, args.test_size, seed=args.seed)
    rcfg = TaskAwareRewardConfig(gate_mode=args.gate_mode)

    # One TaskContext per (spec, noise) with its own baseline accuracy.
    print(f"[noise-grid] {len(specs)} specs x {len(noises)} noise levels | "
          f"{args.dataset} {args.num_qubits}q test={args.test_size}", flush=True)
    ctxs, originals = {}, {}
    for (mt, reps, ent) in specs:
        fm = make_feature_map(args.num_qubits, mt, reps, ent)
        originals[(mt, reps, ent)] = feature_map_to_qasm(fm)
        for p1 in noises:
            dcfg = DownstreamConfig(num_qubits=args.num_qubits, noise_p1=p1,
                                    seed=args.seed, gpu=False)
            ctx = TaskContext(original=fm, X_train=X_tr, y_train=y_tr,
                              X_test=X_te, y_test=y_te, downstream=dcfg)
            ctx.ensure_baseline()
            ctxs[(mt, reps, ent, p1)] = ctx

    tok = transformers.AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    def gen(m, src, mt):
        msgs = [{"role": "system", "content": _TASK_SYSTEM_PROMPT},
                {"role": "user", "content": _compression_prompt(src, args.num_qubits, mt)}]
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = m.generate(ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    results = {"config": vars(args), "noises": noises, "models": {}}
    for entry in args.model or ["sft:"]:
        label, _, adapter = entry.partition(":")
        m = transformers.AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.bfloat16, device_map="cuda:0")
        if adapter:
            m = PeftModel.from_pretrained(m, adapter)
        m.eval()
        # per noise level: accumulate comp%, acc, retention over specs
        by_noise = {p1: {"comp": [], "acc": [], "ret": []} for p1 in noises}
        for (mt, reps, ent) in specs:
            cand = gen(m, originals[(mt, reps, ent)], mt)
            for p1 in noises:
                ctx = ctxs[(mt, reps, ent, p1)]
                r = score_candidate(cand, ctx, rcfg)
                if r["valid"] != 1.0:
                    continue
                by_noise[p1]["comp"].append(r["compression_gain"])
                if r["candidate_accuracy"] is not None:
                    by_noise[p1]["acc"].append(r["candidate_accuracy"])
                    if ctx.baseline_accuracy:
                        by_noise[p1]["ret"].append(
                            r["candidate_accuracy"] / ctx.baseline_accuracy)
        summ = {}
        for p1 in noises:
            d = by_noise[p1]
            summ[p1] = {
                "mean_comp_pct": float(100 * np.mean(d["comp"])) if d["comp"] else None,
                "mean_acc": float(np.mean(d["acc"])) if d["acc"] else None,
                "mean_retention": float(np.mean(d["ret"])) if d["ret"] else None,
            }
        results["models"][label] = summ
        del m
        torch.cuda.empty_cache()
        print(f"[{label}] " + " | ".join(
            f"p1={p1}: comp={summ[p1]['mean_comp_pct']:.1f}% "
            f"acc={summ[p1]['mean_acc']:.3f} ret={summ[p1]['mean_retention']:.3f}"
            for p1 in noises), flush=True)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))

    # Console table: retention vs noise per model.
    print(f"\n=== RETENTION vs NOISE ({args.dataset}) ===")
    hdr = "model     " + "".join(f"  p1={p:<5}" for p in noises)
    print(hdr); print("-" * len(hdr))
    for label, summ in results["models"].items():
        row = f"{label:9s} " + "".join(
            f"  {('n/a' if summ[p]['mean_retention'] is None else format(summ[p]['mean_retention'],'.3f')):>7s}"
            for p in noises)
        print(row)
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
