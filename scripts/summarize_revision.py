#!/usr/bin/env python
"""Build the machine-readable summary and per-seed tables for the revision campaign.

Reads whatever `results/corrected/` holds and skips sections whose inputs are absent, so
it can be run at any point of the campaign. Nothing here reruns a simulation. Writes
`results/corrected/summary.json` and `results/corrected/summary_tables.md`.

Statistics policy (frozen_config.json): every policy seed is reported; the primary
confirmatory comparison is lr 5e-6 against the hand-designed linear arm; paired Wilcoxon
per noise level across the five DATA splits, Holm across levels; with five splits the
per-level floor is 0.0625 and nothing below it is called significant; an effect is
called consistent only if its sign holds in at least four of five policy seeds.
"""
from __future__ import annotations
import glob, json, re, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "corrected"
LRS = [("lr5", "5e-6"), ("lr75", "7.5e-6"), ("lr10", "1e-5")]
SEEDS = [42, 43, 44, 45, 46]
DS = {"IoT": "IoT_Original_Distribution.csv", "UNSW": "UNSW_NB15.csv", "Bot": "UNSW_2018_IoT_Botnet_Final_10_Best.csv"}
DSNAME = {"IoT": "IoTID20", "UNSW": "UNSW-NB15", "Bot": "Bot-IoT"}
RULES = ["QVE3", "QWE3", "NWE3"]

def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None

def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [None]*len(ps); m = len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * ps[i]); running = max(running, adj); out[i] = running
    return out

S = {"runs": {}, "structure": {}, "downstream": {}, "comparison_vs_linear": {}, "pre_vs_post": {},
     "rescoring": {}, "c_sweep": {}, "widths": {}, "tau_widths": {}, "queue": {}}
md = []

# ---------------------------------------------------------------- queue / run status
st = (OUT / "queue_state.txt").read_text().splitlines() if (OUT / "queue_state.txt").exists() else []
for lr, _ in LRS:
    for s in SEEDS:
        n = f"corr_{lr}_s{s}"; rec = {"lr": dict(LRS)[lr], "seed": s, "status": "not started"}
        if any(l.startswith(f"{n} DONE") for l in st): rec["status"] = "done"
        elif any(l.startswith(f"{n} FAILED") for l in st): rec["status"] = "FAILED"
        elif any(l.startswith(f"{n} LAUNCHED") for l in st): rec["status"] = "running"
        rec["attempts"] = sum(1 for l in st if l.startswith(f"{n} LAUNCHED"))
        ts = load(ROOT / f"models/{n}/checkpoint-250/trainer_state.json")
        if ts:
            h = ts["log_history"]; tr = [e for e in h if "rewards/task_aware_grpo_reward/mean" in e]
            rec["final_task_reward"] = tr[-1]["rewards/task_aware_grpo_reward/mean"] if tr else None
            rec["last10_task_reward_mean"] = float(np.mean([e["rewards/task_aware_grpo_reward/mean"] for e in tr[-3:]])) if tr else None
            rt = [e.get("train_runtime") for e in h if "train_runtime" in e]
            if rt:
                rec["train_runtime_min"] = round(rt[-1] / 60, 1)
            else:
                # the final runtime entry is logged after the last checkpoint; fall back
                # to wall time from the run directory's creation to the last checkpoint
                import os
                d = ROOT / f"models/{n}"
                try:
                    rec["train_runtime_min"] = round((os.stat(d / "checkpoint-250/trainer_state.json").st_mtime - os.stat(d).st_ctime) / 60, 1)
                    rec["train_runtime_note"] = "wall time, directory creation to final checkpoint"
                except OSError:
                    rec["train_runtime_min"] = None
            rec["ended_invalid"] = bool(tr and tr[-1]["rewards/task_aware_grpo_reward/mean"] <= -0.29)
        rl = ROOT / f"logs/{n}.rewards.jsonl"
        if rl.exists():
            rows = [json.loads(l) for l in rl.read_text().splitlines() if l.strip()]
            if rows:
                rec["reward_evaluations"] = len(rows)
                rec["frac_valid"] = float(np.mean([r["valid"] for r in rows]))
                eff = [r["effective"] for r in rows if r.get("effective") is not None]
                rec["frac_effective_of_structurally_valid"] = float(np.mean(eff)) if eff else None
                rec["frac_absolute_fallback"] = float(np.mean([r.get("reference_kind") == "absolute_fallback" for r in rows]))
                last = rows[-200:]
                rec["last200_mean_reward"] = float(np.mean([r["reward"] for r in last]))
        S["runs"][n] = rec
S["queue"] = {"done": [l.split()[0] for l in st if " DONE" in l], "failed": [l.split()[0] for l in st if " FAILED" in l]}

# ---------------------------------------------------------------- structure and Eq. 9 audit
struct = load(OUT / "structure.json") or {}
eff = load(OUT / "effective_params.json") or {}
for n, v in struct.items():
    if "error" in v: S["structure"][n] = {"error": v["error"]}; continue
    e = [eff.get(f"{n}__{t}", {}).get("effective") for t in ("Z_1_full", "ZZ_2_full", "Pauli_1_full")]
    S["structure"][n] = {"per_member_2q": v["per_member_2q"], "total_2q": v["total_2q"], "reduction_pct": round(v["reduction_pct"], 1),
                         "distinct_members": v["distinct_members"], "all_valid": v["all_valid"],
                         "effective_per_member": e, "effective_total": sum(x for x in e if x is not None) if all(x is not None for x in e) else None}
if S["structure"]:
    md.append("## Table S1. Corrected policies: committee structure and Eq. 9 audit\n")
    md.append("| Run | lr | seed | g2 (Z/ZZ/Pauli) | total g2 | reduction | distinct | effective / 18 | final task reward | runtime (min) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for lr, lrv in LRS:
        for s in SEEDS:
            n = f"corr_{lr}_s{s}"; v = S["structure"].get(n); r = S["runs"].get(n, {})
            if not v: md.append(f"| {n} | {lrv} | {s} | ({r.get('status','?')}) | | | | | | |"); continue
            if "error" in v: md.append(f"| {n} | {lrv} | {s} | EMIT ERROR | | | | | | |"); continue
            md.append(f"| {n} | {lrv} | {s} | {'/'.join(map(str,v['per_member_2q']))} | {v['total_2q']} | {v['reduction_pct']}% | {v['distinct_members']} | {v['effective_total']} | {r.get('final_task_reward') if r.get('final_task_reward') is None else round(r['final_task_reward'],2)} | {r.get('train_runtime_min','')} |")
    md.append("")
    # per-lr aggregates
    md.append("| lr | runs emitted | compressed at all | all three distinct | 18/18 effective | mean total g2 | mean reduction |")
    md.append("|---|---|---|---|---|---|---|")
    for lr, lrv in LRS:
        vs = [S["structure"][f"corr_{lr}_s{s}"] for s in SEEDS if f"corr_{lr}_s{s}" in S["structure"] and "error" not in S["structure"][f"corr_{lr}_s{s}"]]
        if not vs: continue
        md.append(f"| {lrv} | {len(vs)} | {sum(v['total_2q']<90 for v in vs)} | {sum(v['distinct_members']==3 for v in vs)} | {sum(v['effective_total']==18 for v in vs)} | {np.mean([v['total_2q'] for v in vs]):.1f} | {np.mean([v['reduction_pct'] for v in vs]):.1f}% |")
    md.append("")

# ---------------------------------------------------------------- downstream (matched protocol)
def matched(tag, kind):
    return load(OUT / f"compression_matched_{kind}_{tag}.json")
for tag, dsfile in DS.items():
    for kind in ("corrected", "precorrection"):
        b = matched(tag, kind)
        if not b: continue
        bn = b["datasets"][dsfile]["by_noise"]; noises = list(bn)
        S["downstream"].setdefault(kind, {})[tag] = {"noises": noises, "arms": {}}
        for arm in bn[noises[0]]:
            S["downstream"][kind][tag]["arms"][arm] = {nk: {r: bn[nk][arm]["fusion"][r]["mcc_seeds"] for r in RULES} for nk in noises}
            S["downstream"][kind][tag]["arms"][arm]["meta"] = b["arm_metrics"].get(arm, {})

def arm_series(kind, tag, arm, rule):
    d = S["downstream"].get(kind, {}).get(tag, {}); a = d.get("arms", {}).get(arm)
    return None if not a else {nk: a[nk][rule] for nk in d["noises"]}

if S["downstream"].get("corrected"):
    for rule in RULES:
        md.append(f"## Table S2 ({rule}). Downstream MCC of every corrected policy, matched protocol (200/400, five data splits), mean over splits\n")
        for tag in DS:
            d = S["downstream"]["corrected"].get(tag)
            if not d: continue
            md.append(f"**{DSNAME[tag]}**\n")
            md.append("| Arm | " + " | ".join(f"p1={nk}" for nk in d["noises"]) + " |"); md.append("|---|" + "---|"*len(d["noises"]))
            for arm in ["uncompressed", "l3", "linear", "zonly", "sft"] + [f"corr_{lr}_s{s}" for lr,_ in LRS for s in SEEDS]:
                a = d["arms"].get(arm)
                if not a: continue
                md.append(f"| {arm} | " + " | ".join(f"{np.mean(a[nk][rule]):.3f}" for nk in d["noises"]) + " |")
            # per-lr mean +- sd over policy seeds
            for lr, lrv in LRS:
                arms = [f"corr_{lr}_s{s}" for s in SEEDS if f"corr_{lr}_s{s}" in d["arms"]]
                if not arms: continue
                cells = []
                for nk in d["noises"]:
                    m = [np.mean(d["arms"][a][nk][rule]) for a in arms]
                    cells.append(f"{np.mean(m):.3f} +/- {np.std(m, ddof=1) if len(m)>1 else 0:.3f} (n={len(m)})")
                md.append(f"| **lr {lrv}, mean over seeds** | " + " | ".join(cells) + " |")
            md.append("")

    # comparison vs the hand-designed linear arm: per seed-arm, per level, paired over data splits
    for tag in DS:
        d = S["downstream"]["corrected"].get(tag)
        if not d or "linear" not in d["arms"]: continue
        for rule in RULES:
            rows = []
            for lr, lrv in LRS:
                for s in SEEDS:
                    arm = f"corr_{lr}_s{s}"
                    if arm not in d["arms"]: continue
                    ps, deltas, wtl = [], [], []
                    for nk in d["noises"]:
                        x = np.array(d["arms"][arm][nk][rule]); y = np.array(d["arms"]["linear"][nk][rule]); dd = x - y
                        deltas.append(float(dd.mean())); nz = dd[np.abs(dd) > 1e-12]
                        ps.append(float(wilcoxon(nz).pvalue) if len(nz) else 1.0)
                        wtl.append((int((dd>1e-12).sum()), int((np.abs(dd)<=1e-12).sum()), int((dd<-1e-12).sum())))
                    rows.append({"arm": arm, "lr": lrv, "seed": s, "mean_delta_by_level": deltas, "p_by_level": ps, "holm_by_level": holm(ps), "wtl_by_level": wtl})
            S["comparison_vs_linear"].setdefault(tag, {})[rule] = rows
            if rows:
                md.append(f"## Table S3 ({DSNAME[tag]}, {rule}). Corrected policy minus hand-designed linear arm, paired over the five data splits\n")
                md.append("| Arm | " + " | ".join(f"d@{nk} (w/t/l, Holm p)" for nk in d["noises"]) + " |"); md.append("|---|" + "---|"*len(d["noises"]))
                for r in rows:
                    md.append(f"| {r['arm']} | " + " | ".join(f"{dl:+.3f} ({w}/{t}/{l}, {hp:.3f})" for dl,(w,t,l),hp in zip(r["mean_delta_by_level"], r["wtl_by_level"], r["holm_by_level"])) + " |")
                # consistency across policy seeds per lr
                for lr, lrv in LRS:
                    rr = [r for r in rows if r["lr"] == lrv]
                    if len(rr) >= 2:
                        cons = []
                        for k, nk in enumerate(d["noises"]):
                            signs = [np.sign(r["mean_delta_by_level"][k]) for r in rr]
                            pos = sum(s > 0 for s in signs); neg = sum(s < 0 for s in signs)
                            cons.append(f"{pos}+/{neg}-/{len(signs)-pos-neg}= of {len(signs)}")
                        md.append(f"| **lr {lrv} sign consistency across seeds** | " + " | ".join(cons) + " |")
                md.append("")

# ---------------------------------------------------------------- pre vs post at the shared seeds
pre = load(OUT / "structure.json"); prev = load(ROOT / "results/replicates_structure.json") or {}
if S["structure"] and prev:
    md.append("## Table S4. Pre-correction versus corrected policies at the same training seeds\n")
    md.append("| lr | seed | pre: total g2 / distinct / effective | corrected: total g2 / distinct / effective |"); md.append("|---|---|---|---|")
    preeff = load(ROOT / "results/effective_params.json") or {}
    for lr, lrv in LRS:
        for s, pn in ((42, f"grpo_fix_{lr}"), (43, f"rep_{lr}_s43")):
            pv = prev.get(pn); cv = S["structure"].get(f"corr_{lr}_s{s}")
            pe = sum(preeff.get(f"{pn}__{t}", {}).get("effective", 0) for t in ("Z_1_full","ZZ_2_full","Pauli_1_full")) if pv else None
            md.append(f"| {lrv} | {s} | {pv['total_2q'] if pv else '?'} / {pv['distinct_members'] if pv else '?'} / {pe} | " + (f"{cv['total_2q']} / {cv['distinct_members']} / {cv['effective_total']}" if cv and 'error' not in cv else "(pending)") + " |")
            S["pre_vs_post"][f"{lr}_s{s}"] = {"pre": pn, "pre_struct": pv, "pre_effective": pe, "post_struct": cv}
    md.append("")

# ---------------------------------------------------------------- rescoring
for kind in ("corrected_objective", "original_objective"):
    b = load(OUT / f"rescoring_corrected_circuits_{kind}.json")
    if b: S["rescoring"][kind] = b
b = load(OUT / "rescoring_precorrection_circuits_corrected_objective.json")
if b: S["rescoring"]["precorrection_circuits_corrected_objective"] = b

# ---------------------------------------------------------------- C sweep (E4)
for tag, dsfile in DS.items():
    b = load(OUT / f"c_sweep_{tag}.json")
    if not b: continue
    bn = b["datasets"][dsfile]["by_noise"]; S["c_sweep"][tag] = {}
    md.append(f"## Table S5 ({DSNAME[tag]}). SVM regularisation sweep on shared Gram matrices, mean MCC over five splits\n")
    md.append("| p1 | member / rule | C=0.1 | C=1 | C=10 | C=100 |"); md.append("|---|---|---|---|---|---|")
    for nk, d in bn.items():
        byc = d.get("by_C", {})
        if not byc: continue
        S["c_sweep"][tag][nk] = byc
        for lab in ("Z1", "ZZ2", "Pauli1"):
            md.append(f"| {nk} | {lab} | " + " | ".join(f"{byc[c]['branch'][lab]['mcc_mean']:.3f}" if c in byc else "" for c in ("0.1","1.0","10.0","100.0")) + " |")
        for r in ("QVE3", "NWE3"):
            md.append(f"| {nk} | {r} | " + " | ".join(f"{byc[c]['fusion'][r]['mcc_mean']:.3f}" if c in byc else "" for c in ("0.1","1.0","10.0","100.0")) + " |")
    md.append("")

# ---------------------------------------------------------------- widths (E3) and tau across widths
for w in (8, 10):
    for tag, dsfile in DS.items():
        b = load(OUT / f"fusion_realistic_{w}q_{tag}.json") if tag != "UNSW" else load(ROOT / f"results/fusion_realistic_{w}q.json")
        if not b: continue
        bn = b["datasets"][dsfile]["by_noise"]; S["widths"].setdefault(str(w), {})[tag] = {}
        for nk, d in bn.items():
            f = d["fusion"]; br = d["branch"]
            S["widths"][str(w)][tag][nk] = {"QVE3": f["QVE3"]["mcc_seeds"], "QWE3": f["QWE3"]["mcc_seeds"],
                                            "NWE3": f["NWE3@0.05"]["mcc_seeds"], "tau": {t: f[f"NWE3@{t}"]["mcc_mean"] for t in ("0.01","0.02","0.05","0.1","0.2")},
                                            "branch": {l: br[l]["mcc"] for l in br}, "kept": d.get("kept_rate", {})}
if S["widths"]:
    md.append("## Table S6. Hardware-realistic family at eight and ten qubits, all datasets, mean MCC over five splits\n")
    md.append("| Width | Dataset | p2 | QVE3 | QWE3 | NWE3 | Z | ZZ | Pauli | tau range giving the NWE3 result |"); md.append("|---|---|---|---|---|---|---|---|---|---|")
    for w in ("8", "10"):
        for tag in DS:
            for nk, d in S["widths"].get(w, {}).get(tag, {}).items():
                same = [t for t, v in d["tau"].items() if abs(v - d["tau"]["0.05"]) < 5e-4]
                md.append(f"| {w} | {DSNAME[tag]} | {nk.split(':')[-1]} | {np.mean(d['QVE3']):.3f} | {np.mean(d['QWE3']):.3f} | {np.mean(d['NWE3']):.3f} | {d['branch'].get('Z1',float('nan')):.3f} | {d['branch'].get('ZZ2',float('nan')):.3f} | {d['branch'].get('Pauli1',float('nan')):.3f} | {', '.join(same)} |")
    md.append("")

(OUT / "summary.json").write_text(json.dumps(S, indent=1, default=float))
(OUT / "summary_tables.md").write_text("# Revision campaign summary tables\n\nGenerated by scripts/summarize_revision.py from results/corrected/. Every number is a mean over the five data splits unless a per-split list is shown.\n\n" + "\n".join(md))
print(f"[written] {OUT/'summary.json'}  {OUT/'summary_tables.md'}")
print("runs:", {k: v["status"] for k, v in S["runs"].items()})
print("sections:", {k: bool(v) for k, v in S.items()})
