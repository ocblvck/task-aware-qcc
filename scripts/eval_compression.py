#!/usr/bin/env python
"""Evaluate task-aware GRPO compression vs the SFT baseline and the original maps.

For each feature-map compression task, we let a model generate a compressed
circuit (greedy) and score it with the SAME two-part reward used in training:
  * validity (parses as a same-#qubit circuit)
  * R1 equivalence (exact, for small circuits)
  * compression (2-qubit / depth reduction vs the original)
  * R2 downstream QSVC accuracy under depolarizing noise (+ retention vs original)

Models are compared at a FIXED evaluation noise level for fairness. The "sft"
baseline is the merged SFT-warmed base with no GRPO adapter; "original" is the
uncompressed feature map (compression 0, equivalence 1, reference accuracy).

Run:
  python scripts/eval_compression.py \
     --base models/grpo_taskaware_v1/merged_sft_base \
     --model sft: --model or:models/grpo_or --model and:models/grpo_and \
     --model or_n05:models/grpo_or_n05 \
     --dataset UNSW_NB15.csv --num-qubits 6 --noise-p1 0.01 \
     --output results/compression_eval_unsw_6q.json
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
        for map_type, reps in maps:
            for ent in ("full", "linear"):
                key = (map_type, reps, ent)
                if key not in seen:
                    seen.add(key)
                    specs.append(key)
    return specs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="models/grpo_taskaware_v1/merged_sft_base")
    ap.add_argument("--model", action="append", default=[],
                    help="label:adapter_path  (empty path => base/SFT baseline). Repeatable.")
    ap.add_argument("--dataset", default="UNSW_NB15.csv")
    ap.add_argument("--data-dir", default="/home/chibuike/quantum-ml-iot-nid")
    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--train-size", type=int, default=16)
    ap.add_argument("--test-size", type=int, default=8)
    ap.add_argument("--noise-p1", type=float, default=0.01)
    ap.add_argument("--gate-mode", default="or", choices=["and", "or"])
    ap.add_argument("--max-new-tokens", type=int, default=2600)
    ap.add_argument("--samples", type=int, default=1,
                    help="Candidates per spec. >1 => sampling + best-of-N (by reward).")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Sampling temperature when --samples>1.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/compression_eval.json")
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

    specs = dedup_specs()
    dpath = str(Path(args.data_dir) / args.dataset)
    X_tr, X_te, y_tr, y_te = load_split(
        dpath, args.num_qubits, args.train_size, args.test_size, seed=args.seed
    )
    dcfg = DownstreamConfig(num_qubits=args.num_qubits, noise_p1=args.noise_p1,
                            seed=args.seed, gpu=False)
    rcfg = TaskAwareRewardConfig(gate_mode=args.gate_mode)

    # Pre-build a TaskContext (original map + baseline accuracy) per spec.
    print(f"[eval] {len(specs)} specs | {args.dataset} {args.num_qubits}q | "
          f"noise_p1={args.noise_p1} | gate_mode={args.gate_mode}", flush=True)
    contexts, originals = {}, {}
    for (mt, reps, ent) in specs:
        fm = make_feature_map(args.num_qubits, mt, reps, ent)
        ctx = TaskContext(original=fm, X_train=X_tr, y_train=y_tr,
                          X_test=X_te, y_test=y_te, downstream=dcfg)
        ctx.ensure_baseline()
        contexts[(mt, reps, ent)] = ctx
        originals[(mt, reps, ent)] = feature_map_to_qasm(fm)

    tok = transformers.AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    def load_model(adapter):
        m = transformers.AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.bfloat16, device_map="cuda:0")
        if adapter:
            m = PeftModel.from_pretrained(m, adapter)
        m.eval()
        return m

    def generate(m, src, mt):
        msgs = [{"role": "system", "content": _TASK_SYSTEM_PROMPT},
                {"role": "user", "content": _compression_prompt(src, args.num_qubits, mt)}]
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt").to("cuda:0")
        am = torch.ones_like(ids)
        gens = []
        n = max(1, args.samples)
        sample = n > 1
        with torch.no_grad():
            for _ in range(n):
                kw = dict(do_sample=True, temperature=args.temperature,
                          top_p=0.95, top_k=0) if sample else dict(do_sample=False)
                out = m.generate(ids, attention_mask=am, max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tok.eos_token_id, **kw)
                gens.append(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
        return gens

    model_specs = args.model or ["sft:"]
    results = {"config": vars(args), "baseline_by_spec": {}, "models": {}}
    for (mt, reps, ent), ctx in contexts.items():
        results["baseline_by_spec"][f"{mt}_{reps}_{ent}"] = {
            "orig_metrics": circuit_metrics(ctx.original),
            "baseline_accuracy": ctx.baseline_accuracy,
        }

    for entry in model_specs:
        label, _, adapter = entry.partition(":")
        print(f"\n[model] {label} (adapter={adapter or 'NONE/base'})", flush=True)
        m = load_model(adapter or None)
        rows, comps, equivs, accs, rets, rewards, valids = [], [], [], [], [], [], []
        for (mt, reps, ent), ctx in contexts.items():
            cands = generate(m, originals[(mt, reps, ent)], mt)
            scored = [score_candidate(g, ctx, rcfg) for g in cands]
            # best-of-N: pick the highest-reward candidate for this spec.
            res = max(scored, key=lambda r: r["reward"])
            valid = res["valid"] == 1.0
            valids.append(valid)
            acc = res["candidate_accuracy"]
            ret = (acc / ctx.baseline_accuracy) if (acc and ctx.baseline_accuracy) else None
            if valid:
                comps.append(res["compression_gain"]); rewards.append(res["reward"])
                if res["equiv"] is not None: equivs.append(res["equiv"])
                if acc is not None: accs.append(acc)
                if ret is not None: rets.append(ret)
            rows.append({"spec": f"{mt}_{reps}_{ent}", **res})
            print(f"  {mt:6s} {ent:6s} valid={valid} comp%={100*res['compression_gain']:6.1f} "
                  f"equiv={res['equiv']} acc={acc} reward={res['reward']:.3f}", flush=True)
        del m; torch.cuda.empty_cache()
        summary = {
            "validity_rate": float(np.mean(valids)),
            "mean_compression_pct": float(100 * np.mean(comps)) if comps else None,
            "mean_equivalence": float(np.mean(equivs)) if equivs else None,
            "mean_accuracy": float(np.mean(accs)) if accs else None,
            "mean_accuracy_retention": float(np.mean(rets)) if rets else None,
            "mean_reward": float(np.mean(rewards)) if rewards else None,
        }
        results["models"][label] = {"summary": summary, "per_spec": rows}
        print(f"  => {label}: {summary}", flush=True)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))

    # Console comparison table.
    print("\n=== COMPARISON (eval noise_p1={}) ===".format(args.noise_p1))
    hdr = f"{'model':10s} {'valid%':>7s} {'comp%':>7s} {'equiv':>6s} {'acc':>6s} {'retain':>7s} {'reward':>7s}"
    print(hdr); print("-" * len(hdr))
    for label, d in results["models"].items():
        s = d["summary"]
        def f(x, p=2): return "n/a" if x is None else f"{x:.{p}f}"
        print(f"{label:10s} {100*s['validity_rate']:7.1f} {f(s['mean_compression_pct'],1):>7s} "
              f"{f(s['mean_equivalence']):>6s} {f(s['mean_accuracy']):>6s} "
              f"{f(s['mean_accuracy_retention']):>7s} {f(s['mean_reward']):>7s}")
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
