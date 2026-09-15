"""Score the emitted committee circuits through the same reward the policy was trained
against: UNSW-NB15, 6 qubits, 16 train / 8 test from a 4000-record pool, p1=0.01,
accuracy retention. Answers Reviewer 2: did the inert-parameter circuits earn their reward
through genuinely retained accuracy on this tiny subsample?"""
import sys, json
from pathlib import Path
sys.path.insert(0, "/home/chibuike/task-aware-qcc/src")
from qiskit import qasm3
from taqcc.data import load_split
from taqcc.downstream import DownstreamConfig
from taqcc.feature_maps import make_feature_map, circuit_metrics
from taqcc.reward import TaskAwareRewardConfig, TaskContext, score_candidate
CD=Path(__file__).resolve().parents[1] / "results" / "replicate_circuits"
COMMITTEE=[("Z",1,"full"),("ZZ",2,"full"),("Pauli",1,"full")]
out={"protocol":"UNSW_NB15, 6q, train 16 / test 8, pool 4000, p1=0.01, util_metric=accuracy (the training reward loop)","by_seed":{}}
for seed in (43, 42):
    X_tr,X_te,y_tr,y_te=load_split("/home/chibuike/quantum-ml-iot-nid/UNSW_NB15.csv",6,16,8,pool_size=4000,seed=seed)
    dcfg=DownstreamConfig(num_qubits=6, noise_p1=0.01, seed=seed, gpu=False)
    rec={"test_class_counts":[int((y_te==c).sum()) for c in (0,1)],"train_class_counts":[int((y_tr==c).sum()) for c in (0,1)],"models":{}}
    for model in ("rep_lr5_s43","grpo_fix_lr5","grpo_fix_lr75","sft_warmup"):
        rec["models"][model]={}
        for mt,reps,ent in COMMITTEE:
            src=make_feature_map(6,mt,reps,ent)
            ctx=TaskContext(original=src,X_train=X_tr,y_train=y_tr,X_test=X_te,y_test=y_te,downstream=dcfg)
            base=ctx.ensure_baseline()
            cand=qasm3.loads((CD/f"{model}__{mt}_{reps}_{ent}.comp.qasm").read_text())
            r=score_candidate(cand,ctx,TaskAwareRewardConfig())
            m=circuit_metrics(cand)
            rec["models"][model][mt]={"source_acc":round(base,4),"cand_acc":None if r["candidate_accuracy"] is None else round(r["candidate_accuracy"],4),
                "utility":round(r["utility"],4),"equiv":None if r["equiv"] is None else round(r["equiv"],4),"comp_gain":round(r["compression_gain"],4),
                "gate":round(r["gate"],4),"reward":round(r["reward"],4),"two_qubit":m["two_qubit"],"depth":m["depth"]}
            v=rec["models"][model][mt]
            print(f"seed{seed} {model:<14}{mt:<6} src_acc={v['source_acc']:.3f} cand_acc={v['cand_acc']} util={v['utility']:.3f} equiv={v['equiv']} gamma={v['comp_gain']:.3f} reward={v['reward']:.3f}", flush=True)
    out["by_seed"][str(seed)]=rec
    json.dump(out,open(CD.parents[0] / "reward_loop_rescoring.json","w"),indent=1)
print("[written] results/reward_loop_rescoring.json")
