"""Score the emitted committee circuits through the same reward the policy was trained
against: UNSW-NB15, 6 qubits, 16 train / 8 test from a 4000-record pool, p1=0.01,
accuracy retention. Answers Reviewer 2: did the inert-parameter circuits earn their reward
through genuinely retained accuracy on this tiny subsample?"""
import os
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, "/home/chibuike/task-aware-qcc/src")
from qiskit import qasm3
from taqcc.io import atomic_write_json
from taqcc.data import load_split
from taqcc.downstream import DownstreamConfig
from taqcc.feature_maps import make_feature_map, circuit_metrics
from taqcc.reward import TaskAwareRewardConfig, TaskContext, score_candidate
ap=argparse.ArgumentParser()
ap.add_argument("--circuits", default=str(Path(__file__).resolve().parents[1] / "results" / "replicate_circuits"))
ap.add_argument("--models", default="rep_lr5_s43,grpo_fix_lr5,grpo_fix_lr75,sft_warmup")
ap.add_argument("--seeds", default="43,42")
ap.add_argument("--train-size", type=int, default=16)
ap.add_argument("--test-size", type=int, default=8)
ap.add_argument("--util-metric", default="accuracy", choices=["accuracy","mcc"])
ap.add_argument("--util-reference", default="clean_source", choices=["clean_source","noisy_source"])
ap.add_argument("--require-effective", action="store_true")
ap.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "results" / "reward_loop_rescoring.json"))
args=ap.parse_args()
CD=Path(args.circuits)
COMMITTEE=[("Z",1,"full"),("ZZ",2,"full"),("Pauli",1,"full")]
cfg=TaskAwareRewardConfig(util_metric=args.util_metric, util_reference=args.util_reference, require_effective=args.require_effective)
out={"protocol":f"UNSW_NB15, 6q, train {args.train_size} / test {args.test_size}, pool 4000, p1=0.01, util_metric={args.util_metric}, reference={args.util_reference}, require_effective={args.require_effective}","by_seed":{}}
for seed in [int(s) for s in args.seeds.split(",")]:
    X_tr,X_te,y_tr,y_te=load_split(os.path.join(os.environ.get("TAQCC_DATA_DIR", "data"), "UNSW_NB15.csv"),6,args.train_size,args.test_size,pool_size=4000,seed=seed)
    dcfg=DownstreamConfig(num_qubits=6, noise_p1=0.01, seed=seed, gpu=True)
    rec={"test_class_counts":[int((y_te==c).sum()) for c in (0,1)],"train_class_counts":[int((y_tr==c).sum()) for c in (0,1)],"models":{}}
    for model in [m for m in args.models.split(",") if m]:
        rec["models"][model]={}
        for mt,reps,ent in COMMITTEE:
            src=make_feature_map(6,mt,reps,ent)
            ctx=TaskContext(original=src,X_train=X_tr,y_train=y_tr,X_test=X_te,y_test=y_te,downstream=dcfg)
            base=ctx.ensure_baseline()
            cand=qasm3.loads((CD/f"{model}__{mt}_{reps}_{ent}.comp.qasm").read_text())
            r=score_candidate(cand,ctx,cfg)
            m=circuit_metrics(cand)
            rec["models"][model][mt]={"source_acc":round(base,4),"source_mcc_noisy":None if ctx.baseline_mcc is None else round(ctx.baseline_mcc,4),
                "cand_acc":None if r["candidate_accuracy"] is None else round(r["candidate_accuracy"],4),
                "cand_mcc":None if r.get("candidate_mcc") is None else round(r["candidate_mcc"],4),
                "reference_mcc":r.get("reference_mcc"),"reference_kind":r.get("reference_kind"),
                "valid":r["valid"],"effective":r.get("effective"),
                "utility":round(r["utility"],4),"equiv":None if r["equiv"] is None else round(r["equiv"],4),"comp_gain":round(r["compression_gain"],4),
                "gate":round(r["gate"],4),"reward":round(r["reward"],4),"two_qubit":m["two_qubit"],"depth":m["depth"]}
            v=rec["models"][model][mt]
            print(f"seed{seed} {model:<14}{mt:<6} valid={v['valid']} eff={v['effective']} src_acc={v['source_acc']:.3f} cand_acc={v['cand_acc']} cand_mcc={v['cand_mcc']} ref={v['reference_kind']}:{v['reference_mcc']} util={v['utility']:.3f} gamma={v['comp_gain']:.3f} reward={v['reward']:.3f}", flush=True)
    out["by_seed"][str(seed)]=rec
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, out, indent=1)
print(f"[written] {args.output}")
