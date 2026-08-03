#!/usr/bin/env python
"""Compressed quantum ENSEMBLE evaluation (QVE / QWE) under depolarizing noise.

Addresses the conference paper's two-member-vote limitation by evaluating genuine
odd-committee ensembles (>=3 heterogeneous feature maps). For each ensemble we
compare the ORIGINAL (uncompressed) component maps against their lr5-COMPRESSED
versions, measuring ensemble accuracy retention and the ensemble-level two-qubit
compression. The Z branch (no 2q gates) is kept; the entangling ZZ/Pauli/Custom
branches — which supply ensemble diversity but are deep/noisy — are compressed.

Ensembles (component maps = (map_type, reps, entanglement)):
  QVE3  hard majority: Z, ZZ(full), Pauli(full)                  [3 -> genuine vote]
  QVE5  hard majority: + Custom(full), ZZ(linear)                [5 -> larger committee]
  QWE3  weighted vote: Z, ZZ(full), Pauli(full)                  [3, validation-weighted]

Run:
  python scripts/eval_ensemble_compression.py --base models/sft_compress_e2_merged \
     --model models/grpo_e2_lr5 \
     --datasets IoT_Original_Distribution.csv,UNSW_NB15.csv,UNSW_2018_IoT_Botnet_Final_10_Best.csv \
     --num-qubits 6 --train-size 32 --test-size 32 --noise-p1 0.01 \
     --output results/ensemble_compression_lr5.json
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

# Component maps per ensemble: (map_type, reps, entanglement).
ENSEMBLES = {
    "QVE3": ("majority", [("Z", 1, "full"), ("ZZ", 2, "full"), ("Pauli", 1, "full")]),
    "QVE5": ("majority", [("Z", 1, "full"), ("ZZ", 2, "full"), ("Pauli", 1, "full"),
                          ("Custom", 1, "full"), ("ZZ", 2, "linear")]),
    "QWE3": ("weighted", [("Z", 1, "full"), ("ZZ", 2, "full"), ("Pauli", 1, "full")]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True, help="GRPO adapter path (e.g. lr5)")
    ap.add_argument("--datasets", default="IoT_Original_Distribution.csv,UNSW_NB15.csv,"
                                          "UNSW_2018_IoT_Botnet_Final_10_Best.csv")
    ap.add_argument("--data-dir", default="/home/chibuike/quantum-ml-iot-nid")
    ap.add_argument("--num-qubits", type=int, default=6)
    ap.add_argument("--train-size", type=int, default=32)
    ap.add_argument("--test-size", type=int, default=32)
    ap.add_argument("--noise-p1", type=float, default=0.01)
    ap.add_argument("--noise-grid", default="",
                    help="Comma list of noise levels; overrides --noise-p1 when set.")
    ap.add_argument("--max-new-tokens", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/ensemble_compression.json")
    args = ap.parse_args()

    import torch
    import transformers
    from peft import PeftModel

    from taqcc.data import load_split
    from taqcc.downstream import (
        DownstreamConfig, downstream_accuracy_ensemble,
        downstream_accuracy_weighted_ensemble,
    )
    from taqcc.feature_maps import make_feature_map, circuit_metrics
    from taqcc.grpo_integration import (
        _compression_prompt, _TASK_SYSTEM_PROMPT, feature_map_to_qasm,
    )
    from taqcc.qasm_adapter import parse_candidate

    # Unique component maps across all ensembles.
    comps = []
    for _, maps in ENSEMBLES.values():
        for m in maps:
            if m not in comps:
                comps.append(m)

    tok = transformers.AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(model, args.model)
    model.eval()

    def compress(fm, mt):
        src = feature_map_to_qasm(fm)
        ids = tok.apply_chat_template(
            [{"role": "system", "content": _TASK_SYSTEM_PROMPT},
             {"role": "user", "content": _compression_prompt(src, args.num_qubits, mt)}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        c = parse_candidate(gen)
        # Fall back to the original map if the compressed circuit is unusable.
        if c is None or c.num_qubits != args.num_qubits or c.num_parameters != args.num_qubits:
            return fm, False
        return c, True

    # Generate compressed component maps ONCE (data-independent).
    orig_map, comp_map, comp_meta = {}, {}, {}
    for (mt, reps, ent) in comps:
        fm = make_feature_map(args.num_qubits, mt, reps, ent)
        cc, ok = compress(fm, mt)
        orig_map[(mt, reps, ent)] = fm
        comp_map[(mt, reps, ent)] = cc
        o2, c2 = circuit_metrics(fm)["two_qubit"], circuit_metrics(cc)["two_qubit"]
        comp_meta[(mt, reps, ent)] = {"orig_2q": o2, "comp_2q": c2, "compressed_ok": ok}
        print(f"[compress] {mt:6s} {ent:6s}: 2q {o2}->{c2} ok={ok}", flush=True)
    del model
    torch.cuda.empty_cache()

    results = {"config": vars(args), "component_metrics":
               {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in comp_meta.items()}, "datasets": {}}

    noises = ([float(x) for x in args.noise_grid.split(",")]
              if args.noise_grid else [args.noise_p1])
    for ds in args.datasets.split(","):
        ds = ds.strip()
        X_tr, X_te, y_tr, y_te = load_split(
            str(Path(args.data_dir) / ds), args.num_qubits,
            args.train_size, args.test_size, seed=args.seed)
        print(f"\n[dataset] {ds} | noise grid {noises}", flush=True)
        rows = {}
        for name, (mode, maps) in ENSEMBLES.items():
            orig = [orig_map[m] for m in maps]
            comp = [comp_map[m] for m in maps]
            fn = (downstream_accuracy_weighted_ensemble if mode == "weighted"
                  else downstream_accuracy_ensemble)
            tot_o = sum(comp_meta[m]["orig_2q"] for m in maps)
            tot_c = sum(comp_meta[m]["comp_2q"] for m in maps)
            comp_pct = 100 * (tot_o - tot_c) / tot_o if tot_o else 0.0
            per_noise = {}
            for p1 in noises:
                dcfg = DownstreamConfig(num_qubits=args.num_qubits, noise_p1=p1,
                                        seed=args.seed, gpu=False)
                r_o = fn(orig, X_tr, y_tr, X_te, y_te, dcfg)
                r_c = fn(comp, X_tr, y_tr, X_te, y_te, dcfg)
                ret = r_c["accuracy"] / r_o["accuracy"] if r_o["accuracy"] else None
                per_noise[p1] = {"orig_acc": r_o["accuracy"], "comp_acc": r_c["accuracy"],
                                 "orig_mcc": r_o["mcc"], "comp_mcc": r_c["mcc"],
                                 "retention": ret}
            rows[name] = {"n_maps": len(maps), "mode": mode,
                          "ensemble_2q_orig": tot_o, "ensemble_2q_comp": tot_c,
                          "ensemble_comp_pct": comp_pct, "by_noise": per_noise}
            ret_str = " ".join(f"p{p}:{per_noise[p]['retention']:.2f}" for p in noises)
            mcc_str = " ".join(f"p{p}:{per_noise[p]['orig_mcc']:.2f}->{per_noise[p]['comp_mcc']:.2f}" for p in noises)
            print(f"  {name:5s} ({mode:8s},{len(maps)}m) 2q {tot_o}->{tot_c} "
                  f"({comp_pct:.1f}%) | retention {ret_str}", flush=True)
            print(f"        MCC orig->comp {mcc_str}", flush=True)
        results["datasets"][ds] = rows

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
