#!/usr/bin/env python
"""Reproducible, seeded ensemble-compression evaluation with accuracy + MCC.

Fixes two robustness problems in the small-sample eval:
  (1) Non-determinism: the lr5-compressed circuits are generated ONCE (greedy,
      seeded) and cached to disk as OpenQASM; every evaluation loads the fixed
      circuits, so results are fully reproducible.
  (2) Small-sample fragility: we evaluate on larger test sets averaged over
      several data-split seeds and report mean +/- std of accuracy, MCC, and
      accuracy retention, across the depolarizing-noise grid, for the genuine
      >=3-map ensembles QVE3 (majority), QVE5 (majority), QWE3 (weighted).

Run:
  python scripts/eval_robust_ensemble.py --base models/sft_compress_e2_merged \
     --model models/grpo_e2_lr5 --seeds 0,1,2,3,4 \
     --train-size 48 --test-size 200 --noise-grid 0.0,0.01,0.03,0.05,0.1 \
     --output results/robust_ensemble_lr5.json
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

ENSEMBLES = {
    "QVE3": ("majority", [("Z", 1, "full"), ("ZZ", 2, "full"), ("Pauli", 1, "full")]),
    "QVE5": ("majority", [("Z", 1, "full"), ("ZZ", 2, "full"), ("Pauli", 1, "full"),
                          ("Custom", 1, "full"), ("ZZ", 2, "linear")]),
    "QWE3": ("weighted", [("Z", 1, "full"), ("ZZ", 2, "full"), ("Pauli", 1, "full")]),
}


def cache_compressed_maps(base, model_path, num_qubits, cache_dir, max_new_tokens=1800):
    """Generate lr5-compressed component maps ONCE (greedy, seeded) and cache QASM."""
    import torch
    import transformers
    from peft import PeftModel
    from taqcc.feature_maps import make_feature_map, circuit_metrics
    from taqcc.grpo_integration import (
        _compression_prompt, _TASK_SYSTEM_PROMPT, feature_map_to_qasm,
    )
    from taqcc.qasm_adapter import parse_candidate

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    comps = []
    for _, maps in ENSEMBLES.values():
        for m in maps:
            if m not in comps:
                comps.append(m)

    meta_path = cache_dir / "compressed_meta.json"
    if meta_path.exists():
        print(f"[cache] reusing cached circuits in {cache_dir}", flush=True)
        return json.loads(meta_path.read_text())

    torch.manual_seed(0)
    tok = transformers.AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    meta = {}
    for (mt, reps, ent) in comps:
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
        key = f"{mt}_{reps}_{ent}"
        ok = cc is not None and cc.num_qubits == num_qubits and cc.num_parameters == num_qubits
        # Cache the source (original) and the compressed QASM (or original as fallback).
        (cache_dir / f"{key}.orig.qasm").write_text(src)
        (cache_dir / f"{key}.comp.qasm").write_text(gen if ok else src)
        meta[key] = {"orig_2q": circuit_metrics(fm)["two_qubit"],
                     "comp_2q": circuit_metrics(cc)["two_qubit"] if ok else circuit_metrics(fm)["two_qubit"],
                     "compressed_ok": bool(ok)}
        print(f"[cache] {key}: 2q {meta[key]['orig_2q']}->{meta[key]['comp_2q']} ok={ok}", flush=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    del model
    torch.cuda.empty_cache()
    return meta


def load_cached(cache_dir, key, kind):
    from taqcc.qasm_adapter import parse_candidate
    from qiskit import qasm3
    txt = (Path(cache_dir) / f"{key}.{kind}.qasm").read_text()
    try:
        return qasm3.loads(txt)
    except Exception:
        return parse_candidate(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", default="IoT_Original_Distribution.csv,UNSW_NB15.csv,"
                                          "UNSW_2018_IoT_Botnet_Final_10_Best.csv")
    ap.add_argument("--data-dir", default="/home/chibuike/quantum-ml-iot-nid")
    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--train-size", type=int, default=48)
    ap.add_argument("--test-size", type=int, default=200)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--noise-grid", default="0.0,0.01,0.03,0.05,0.1")
    ap.add_argument("--cache-dir", default="results/compressed_maps")
    ap.add_argument("--output", default="results/robust_ensemble_lr5.json")
    args = ap.parse_args()

    from taqcc.data import load_split
    from taqcc.downstream import (
        DownstreamConfig, downstream_accuracy_ensemble,
        downstream_accuracy_weighted_ensemble,
    )

    meta = cache_compressed_maps(args.base, args.model, args.num_qubits, args.cache_dir)
    # Load fixed circuits once.
    keys = {m for _, maps in ENSEMBLES.values() for m in maps}
    orig_c = {m: load_cached(args.cache_dir, f"{m[0]}_{m[1]}_{m[2]}", "orig") for m in keys}
    comp_c = {m: load_cached(args.cache_dir, f"{m[0]}_{m[1]}_{m[2]}", "comp") for m in keys}

    seeds = [int(s) for s in args.seeds.split(",")]
    noises = [float(x) for x in args.noise_grid.split(",")]
    results = {"config": vars(args), "component_metrics": meta, "datasets": {}}

    for ds in args.datasets.split(","):
        ds = ds.strip()
        print(f"\n[dataset] {ds}", flush=True)
        ens_out = {}
        for name, (mode, maps) in ENSEMBLES.items():
            orig = [orig_c[m] for m in maps]
            comp = [comp_c[m] for m in maps]
            fn = (downstream_accuracy_weighted_ensemble if mode == "weighted"
                  else downstream_accuracy_ensemble)
            tot_o = sum(meta[f"{m[0]}_{m[1]}_{m[2]}"]["orig_2q"] for m in maps)
            tot_c = sum(meta[f"{m[0]}_{m[1]}_{m[2]}"]["comp_2q"] for m in maps)
            comp_pct = 100 * (tot_o - tot_c) / tot_o if tot_o else 0.0
            by_noise = {}
            for p1 in noises:
                oa, ca, om, cm = [], [], [], []
                for sd in seeds:
                    X_tr, X_te, y_tr, y_te = load_split(
                        str(Path(args.data_dir) / ds), args.num_qubits,
                        args.train_size, args.test_size, seed=sd)
                    dcfg = DownstreamConfig(num_qubits=args.num_qubits, noise_p1=p1,
                                            seed=sd, gpu=False)
                    r_o = fn(orig, X_tr, y_tr, X_te, y_te, dcfg)
                    r_c = fn(comp, X_tr, y_tr, X_te, y_te, dcfg)
                    oa.append(r_o["accuracy"]); ca.append(r_c["accuracy"])
                    om.append(r_o["mcc"]); cm.append(r_c["mcc"])
                by_noise[p1] = {
                    "orig_acc": [float(np.mean(oa)), float(np.std(oa))],
                    "comp_acc": [float(np.mean(ca)), float(np.std(ca))],
                    "orig_mcc": [float(np.mean(om)), float(np.std(om))],
                    "comp_mcc": [float(np.mean(cm)), float(np.std(cm))],
                }
            ens_out[name] = {"n_maps": len(maps), "mode": mode,
                             "ensemble_comp_pct": comp_pct, "by_noise": by_noise}
            line = " ".join(
                f"p{p}:acc {by_noise[p]['orig_acc'][0]:.2f}->{by_noise[p]['comp_acc'][0]:.2f}"
                f" mcc {by_noise[p]['orig_mcc'][0]:.2f}->{by_noise[p]['comp_mcc'][0]:.2f}"
                for p in noises)
            print(f"  {name} ({mode},{len(maps)}m) 2q-red {comp_pct:.1f}% | {line}", flush=True)
        results["datasets"][ds] = ens_out

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
