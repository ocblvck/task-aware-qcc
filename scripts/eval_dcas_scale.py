#!/usr/bin/env python
"""DCAS-scale evaluation: do LLM-compressed QVE/QWE preserve 5-class accuracy?

Reproduces the conference setting (multi-class IoTID20, 10--16 qubits, ideal
statevector kernels, accuracy/MCC/F1) and compares the ORIGINAL feature-map
ensembles against their lr5-COMPRESSED counterparts over multiple seeds. Noiseless
by design, so it extends the DCAS paper without overlapping the noise study.

Run:
  python scripts/eval_dcas_scale.py --base models/sft_compress_e2_merged \
     --model models/grpo_e2_lr5 --dataset IoT_Original_Distribution.csv \
     --num-qubits 10 --train-size 600 --test-size 300 --pool-size 4000 \
     --seeds 42,1,2 --output results/dcas_scale_10q.json
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
    "QSVC": ("single",   [("ZZ", 2, "full")]),
    "QVE3": ("majority", [("Z", 1, "full"), ("ZZ", 2, "full"), ("Pauli", 1, "full")]),
    "QWE3": ("weighted", [("Z", 1, "full"), ("ZZ", 2, "full"), ("Pauli", 1, "full")]),
}


def get_xp():
    try:
        import cupy as cp
        cp.zeros(1)
        return cp, True
    except Exception:
        return np, False


def statevecs(fm, X):
    from qiskit.quantum_info import Statevector
    return np.asarray([Statevector(fm.assign_parameters(x)).data for x in X], dtype=np.complex64)


def gram(Psi_a, Psi_b, xp, gpu):
    A = xp.asarray(Psi_a); B = xp.asarray(Psi_b)
    G = xp.abs(A @ B.conj().T) ** 2
    return xp.asnumpy(G) if gpu else G


def fit_predict(Ktr, Kte, ytr, seed):
    from sklearn.svm import SVC
    svc = SVC(kernel="precomputed", class_weight="balanced", random_state=seed)
    svc.fit(Ktr, ytr)
    return svc.predict(Kte)


def ensemble_predict(mode, comp_preds, comp_valaccs, classes):
    P = np.array(comp_preds)                      # (M, N) integer labels
    if mode == "single" or P.shape[0] == 1:
        return P[0]
    w = np.ones(P.shape[0]) if mode == "majority" else np.array(comp_valaccs)
    w = w / w.sum()
    N = P.shape[1]
    votes = np.zeros((N, len(classes)))
    for m in range(P.shape[0]):
        for c_idx, c in enumerate(classes):
            votes[P[m] == c, c_idx] += w[m]
    return classes[np.argmax(votes, axis=1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="IoT_Original_Distribution.csv")
    ap.add_argument("--data-dir", default="/home/chibuike/quantum-ml-iot-nid")
    ap.add_argument("--num-qubits", type=int, default=10)
    ap.add_argument("--train-size", type=int, default=600)
    ap.add_argument("--test-size", type=int, default=300)
    ap.add_argument("--pool-size", type=int, default=4000)
    ap.add_argument("--seeds", default="42,1,2")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--output", default="results/dcas_scale.json")
    args = ap.parse_args()

    import torch
    import transformers
    from peft import PeftModel
    from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
    from sklearn.model_selection import train_test_split

    from taqcc.data import load_split_multiclass
    from taqcc.feature_maps import make_feature_map, circuit_metrics, is_valid_feature_map
    from taqcc.grpo_integration import (_compression_prompt, _TASK_SYSTEM_PROMPT,
                                        feature_map_to_qasm)
    from taqcc.qasm_adapter import parse_candidate

    nq = args.num_qubits
    comps = []
    for _, maps in ENSEMBLES.values():
        for m in maps:
            if m not in comps:
                comps.append(m)

    # --- 1. generate compressed component maps once (LLM on GPU) ---
    tok = transformers.AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(model, args.model).eval()
    orig_map, comp_map, cost = {}, {}, {}
    for (mt, reps, ent) in comps:
        fm = make_feature_map(nq, mt, reps, ent)
        src = feature_map_to_qasm(fm)
        ids = tok.apply_chat_template(
            [{"role": "system", "content": _TASK_SYSTEM_PROMPT},
             {"role": "user", "content": _compression_prompt(src, nq, mt)}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        c = parse_candidate(gen)
        ok = is_valid_feature_map(c, nq)
        orig_map[(mt, reps, ent)] = fm
        comp_map[(mt, reps, ent)] = c if ok else fm
        mo, mc = circuit_metrics(fm), circuit_metrics(c if ok else fm)
        cost[(mt, reps, ent)] = {"orig_2q": mo["two_qubit"], "comp_2q": mc["two_qubit"],
                                 "orig_depth": mo["depth"], "comp_depth": mc["depth"],
                                 "compressed_ok": ok}
        print(f"[compress] {mt:6s} 2q {mo['two_qubit']}->{mc['two_qubit']} "
              f"depth {mo['depth']}->{mc['depth']} ok={ok}", flush=True)
    del model
    torch.cuda.empty_cache()

    xp, gpu = get_xp()
    print(f"[kernel] backend={'cupy/GPU' if gpu else 'numpy/CPU'}", flush=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    per_seed = {"orig": {k: [] for k in ENSEMBLES}, "comp": {k: [] for k in ENSEMBLES}}
    for seed in seeds:
        Xtr, Xte, ytr, yte = load_split_multiclass(
            str(Path(args.data_dir) / args.dataset), nq,
            args.train_size, args.test_size, pool_size=args.pool_size, seed=seed)
        classes = np.unique(ytr)
        # internal val split for QWE weights
        Xtr2, Xval, ytr2, yval = train_test_split(Xtr, ytr, test_size=0.2,
                                                  stratify=ytr, random_state=seed)
        for variant, fmap in (("orig", orig_map), ("comp", comp_map)):
            comp_pred, comp_valacc = {}, {}
            for key in comps:
                fm = fmap[key]
                Ptr, Pte = statevecs(fm, Xtr), statevecs(fm, Xte)
                Ktr = gram(Ptr, Ptr, xp, gpu); Kte = gram(Pte, Ptr, xp, gpu)
                comp_pred[key] = fit_predict(Ktr, Kte, ytr, seed)
                Ptr2, Pv = statevecs(fm, Xtr2), statevecs(fm, Xval)
                Kt2 = gram(Ptr2, Ptr2, xp, gpu); Kv = gram(Pv, Ptr2, xp, gpu)
                comp_valacc[key] = accuracy_score(yval, fit_predict(Kt2, Kv, ytr2, seed))
            for name, (mode, maps) in ENSEMBLES.items():
                preds = [comp_pred[m] for m in maps]
                vacc = [comp_valacc[m] for m in maps]
                yp = ensemble_predict(mode, preds, vacc, classes)
                per_seed[variant][name].append({
                    "acc": float(accuracy_score(yte, yp)),
                    "mcc": float(matthews_corrcoef(yte, yp)),
                    "f1": float(f1_score(yte, yp, average="weighted")),
                })
        print(f"[seed {seed}] done", flush=True)

    def agg(lst, m):
        v = [r[m] for r in lst]
        return float(np.mean(v)), float(np.std(v))

    results = {"config": vars(args), "component_cost":
               {f"{k[0]}": v for k, v in cost.items()}, "ensembles": {}}
    print(f"\n=== DCAS-scale {args.dataset} {nq}q, 5-class, ideal, seeds={seeds} ===")
    hdr = f"{'ensemble':6s} {'variant':5s} {'acc':>14s} {'mcc':>14s} {'f1':>14s} {'2q':>8s}"
    print(hdr); print("-" * len(hdr))
    for name, (mode, maps) in ENSEMBLES.items():
        tot_o = sum(cost[m]["orig_2q"] for m in maps)
        tot_c = sum(cost[m]["comp_2q"] for m in maps)
        results["ensembles"][name] = {"maps": [f"{m[0]}" for m in maps],
                                      "ensemble_2q_orig": tot_o, "ensemble_2q_comp": tot_c}
        for variant in ("orig", "comp"):
            a = agg(per_seed[variant][name], "acc")
            mc = agg(per_seed[variant][name], "mcc")
            f = agg(per_seed[variant][name], "f1")
            tq = tot_o if variant == "orig" else tot_c
            results["ensembles"][name][variant] = {"acc": a, "mcc": mc, "f1": f}
            print(f"{name:6s} {variant:5s} {a[0]:.3f}+-{a[1]:.3f}  {mc[0]:.3f}+-{mc[1]:.3f}"
                  f"  {f[0]:.3f}+-{f[1]:.3f} {tq:8d}")

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
