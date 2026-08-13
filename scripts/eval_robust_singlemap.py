#!/usr/bin/env python
"""Reproducible, seeded single-map compression evaluation (SFT vs GRPO lr-sweep).

Robust version of the Table-2 lr-sweep. Mirrors eval_robust_ensemble.py:
  (1) Determinism: each model's greedy-compressed circuit per feature-map spec is
      generated ONCE (seeded) and cached to disk as text; every evaluation loads
      the fixed circuits, so results are fully reproducible.
  (2) Small-sample fragility: single-map QSVC is scored on larger test sets
      averaged over several data-split seeds; we report mean +/- std of accuracy,
      MCC, and accuracy retention vs the uncompressed original, plus mean
      two-qubit-gate reduction, at the evaluation noise level(s).

Run:
  python scripts/eval_robust_singlemap.py \
     --base models/sft_compress_e2_merged \
     --model sft: --model lr5:models/grpo_e2_lr5 \
     --model lr75:models/grpo_e2_lr75 --model lr10b:models/grpo_e2_lr10b \
     --datasets UNSW_NB15.csv,UNSW_2018_IoT_Botnet_Final_10_Best.csv \
     --seeds 0,1,2,3,4 --train-size 48 --test-size 160 --noise-grid 0.01 \
     --output results/robust_singlemap.json
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


def cache_model_maps(base, label, adapter, specs, num_qubits, cache_dir,
                     max_new_tokens=2600):
    """Generate greedy-compressed circuits for one model ONCE and cache them."""
    import torch
    import transformers
    from peft import PeftModel
    from taqcc.feature_maps import (
        make_feature_map, circuit_metrics, is_valid_feature_map,
    )
    from taqcc.grpo_integration import (
        _compression_prompt, _TASK_SYSTEM_PROMPT, feature_map_to_qasm,
    )
    from taqcc.qasm_adapter import parse_candidate
    from taqcc.feature_maps import is_valid_feature_map

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / f"{label}_meta.json"
    if meta_path.exists():
        print(f"[cache] reusing {label} circuits in {cache_dir}", flush=True)
        return json.loads(meta_path.read_text())

    torch.manual_seed(0)
    tok = transformers.AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cuda:0")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    meta = {}
    for (mt, reps, ent) in specs:
        key = f"{mt}_{reps}_{ent}"
        fm = make_feature_map(num_qubits, mt, reps, ent)
        src = feature_map_to_qasm(fm)
        ids = tok.apply_chat_template(
            [{"role": "system", "content": _TASK_SYSTEM_PROMPT},
             {"role": "user", "content": _compression_prompt(src, num_qubits, mt)}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        cc = parse_candidate(gen)
        ok = is_valid_feature_map(cc, num_qubits)
        (cache_dir / f"{label}__{key}.comp.qasm").write_text(gen if ok else src)
        (cache_dir / f"{label}__{key}.orig.qasm").write_text(src)
        meta[key] = {"orig_2q": circuit_metrics(fm)["two_qubit"],
                     "comp_2q": circuit_metrics(cc)["two_qubit"] if ok else circuit_metrics(fm)["two_qubit"],
                     "compressed_ok": bool(ok)}
        print(f"[cache:{label}] {key}: 2q {meta[key]['orig_2q']}->{meta[key]['comp_2q']} ok={ok}",
              flush=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    del model
    torch.cuda.empty_cache()
    return meta


def load_cached(cache_dir, label, key, kind):
    from taqcc.qasm_adapter import parse_candidate
    from qiskit import qasm3
    txt = (Path(cache_dir) / f"{label}__{key}.{kind}.qasm").read_text()
    try:
        return qasm3.loads(txt)
    except Exception:
        return parse_candidate(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", action="append", default=[],
                    help="label:adapter_path (empty path => base/SFT). Repeatable.")
    ap.add_argument("--datasets", default="UNSW_NB15.csv,"
                                          "UNSW_2018_IoT_Botnet_Final_10_Best.csv")
    ap.add_argument("--data-dir", default="/home/chibuike/quantum-ml-iot-nid")
    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--train-size", type=int, default=48)
    ap.add_argument("--test-size", type=int, default=160)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--noise-grid", default="0.01")
    ap.add_argument("--cache-dir", default="results/compressed_singlemap")
    ap.add_argument("--output", default="results/robust_singlemap.json")
    args = ap.parse_args()

    from taqcc.data import load_split
    from taqcc.downstream import DownstreamConfig, downstream_accuracy

    specs = dedup_specs()
    models = []
    for spec in args.model:
        label, _, adapter = spec.partition(":")
        models.append((label, adapter))

    # 1) Cache greedy-compressed circuits for every model ONCE.
    metas = {}
    for label, adapter in models:
        metas[label] = cache_model_maps(args.base, label, adapter, specs,
                                        args.num_qubits, args.cache_dir)

    seeds = [int(s) for s in args.seeds.split(",")]
    noises = [float(x) for x in args.noise_grid.split(",")]
    results = {"config": vars(args), "datasets": {}}

    for ds in args.datasets.split(","):
        ds = ds.strip()
        print(f"\n[dataset] {ds}", flush=True)
        model_out = {}
        for label, _ in models:
            meta = metas[label]
            # Mean 2q reduction across specs (skip entanglement-free Z maps: no 2q gates).
            reds = []
            for (mt, reps, ent) in specs:
                o = meta[f"{mt}_{reps}_{ent}"]["orig_2q"]
                c = meta[f"{mt}_{reps}_{ent}"]["comp_2q"]
                if o > 0:
                    reds.append(100 * (o - c) / o)
            comp_pct = float(np.mean(reds)) if reds else 0.0
            by_noise = {}
            for p1 in noises:
                accs, mccs, rets = [], [], []
                for sd in seeds:
                    X_tr, X_te, y_tr, y_te = load_split(
                        str(Path(args.data_dir) / ds), args.num_qubits,
                        args.train_size, args.test_size, seed=sd)
                    dcfg = DownstreamConfig(num_qubits=args.num_qubits, noise_p1=p1,
                                            seed=sd, gpu=False)
                    # Average over specs: compressed acc/mcc and retention vs original.
                    sa, sm, sr = [], [], []
                    for (mt, reps, ent) in specs:
                        key = f"{mt}_{reps}_{ent}"
                        oc = load_cached(args.cache_dir, label, key, "orig")
                        cc = load_cached(args.cache_dir, label, key, "comp")
                        r_o = downstream_accuracy(oc, X_tr, y_tr, X_te, y_te, dcfg)
                        r_c = downstream_accuracy(cc, X_tr, y_tr, X_te, y_te, dcfg)
                        sa.append(r_c["accuracy"]); sm.append(r_c["mcc"])
                        if r_o["accuracy"] > 0:
                            sr.append(r_c["accuracy"] / r_o["accuracy"])
                    accs.append(np.mean(sa)); mccs.append(np.mean(sm)); rets.append(np.mean(sr))
                by_noise[p1] = {
                    "comp_acc": [float(np.mean(accs)), float(np.std(accs))],
                    "comp_mcc": [float(np.mean(mccs)), float(np.std(mccs))],
                    "retention": [float(np.mean(rets)), float(np.std(rets))],
                    "seeds": seeds,
                    "comp_acc_seeds": [float(v) for v in accs],
                    "comp_mcc_seeds": [float(v) for v in mccs],
                    "retention_seeds": [float(v) for v in rets],
                }
            model_out[label] = {"comp_pct": comp_pct, "by_noise": by_noise}
            line = " ".join(
                f"p{p}:acc {by_noise[p]['comp_acc'][0]:.2f}"
                f" mcc {by_noise[p]['comp_mcc'][0]:.2f}"
                f" ret {by_noise[p]['retention'][0]:.2f}" for p in noises)
            print(f"  {label}: comp {comp_pct:.1f}% | {line}", flush=True)
        results["datasets"][ds] = model_out

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
