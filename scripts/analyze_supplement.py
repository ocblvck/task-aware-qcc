#!/usr/bin/env python
"""Analysis of the supplement, written before its results existed (protocol:
results/supplement/PROTOCOL.md).

S1  six-qubit grid on data splits 5 to 14, reported separately from splits 0 to 4.
    Per dataset and noise level: per-split values, mean, sample SD, wins/ties/losses of
    NWE3@0.05 against QVE3 and QWE3, exact two-sided Wilcoxon signed-rank p with tied pairs
    dropped, Holm across the six non-zero levels within a dataset and comparison.
S2  device-derived noise: the same summary per snapshot (five splits; floor p = 0.0625).
S3  structure and downstream MCC of the policies whose reward subsample follows the
    training seed, next to the seed-42-subsample policies at the same rate.

Writes results/supplement/summary_supplement.json and tex tables. Reads JSON only.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from taqcc.io import atomic_write_json
SUP = ROOT / "results/supplement"; TEX = SUP / "tex"; TEX.mkdir(parents=True, exist_ok=True)
DS = {"IoT": ("IoT_Original_Distribution.csv", "IoTID20"), "UNSW": ("UNSW_NB15.csv", "UNSW-NB15"), "Bot": ("UNSW_2018_IoT_Botnet_Final_10_Best.csv", "Bot-IoT")}
ORIG6 = {"IoT": "fusion_v2_IoT_Orig.json", "UNSW": "fusion_v2_UNSW_NB1.json", "Bot": "fusion_v2_UNSW_201.json"}
DEVS = ["fake_casablanca", "fake_jakarta", "fake_lagos", "fake_nairobi", "fake_oslo", "fake_perth"]
EPS = 1e-9

def exact_p(d):
    d = np.asarray(d, float); nz = d[np.abs(d) > EPS]
    if len(nz) == 0: return 1.0
    return float(wilcoxon(nz, zero_method="wilcox", alternative="two-sided", method="exact").pvalue)

def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()

def compare(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return {"mean_delta": float(d.mean()), "median_delta": float(np.median(d)), "wins": int((d > EPS).sum()), "ties": int((np.abs(d) <= EPS).sum()),
            "losses": int((d < -EPS).sum()), "p_exact": exact_p(d), "per_split_delta": d.round(4).tolist()}

def cell(c):
    fu = c["fusion"]; out = {"n_splits": len(fu["QVE3"]["mcc_seeds"])}
    for r in ("QVE3", "QWE3", "NWE3@0.05"):
        v = np.asarray(fu[r]["mcc_seeds"], float); out[r] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "per_split": v.round(4).tolist()}
    out["branch_mean"] = {m: float(b["mcc"]) for m, b in c["branch"].items()}
    out["kept_rate"] = c.get("kept_rate")
    out["tau_sweep_mean"] = {k: float(np.mean(v["mcc_seeds"])) for k, v in fu.items() if k.startswith("NWE3@")}
    out["NWE3_vs_QVE3"] = compare(fu["NWE3@0.05"]["mcc_seeds"], fu["QVE3"]["mcc_seeds"])
    out["NWE3_vs_QWE3"] = compare(fu["NWE3@0.05"]["mcc_seeds"], fu["QWE3"]["mcc_seeds"])
    return out

def s1():
    res = {}
    for tag, (f, name) in DS.items():
        p = SUP / f"fusion_6q_splits5to14_{tag}.json"
        if not p.exists(): continue
        bn = json.load(open(p))["datasets"][f]["by_noise"]; r = {nk: cell(c) for nk, c in bn.items()}
        for cmp_ in ("NWE3_vs_QVE3", "NWE3_vs_QWE3"):
            nz = [nk for nk in r if float(nk) > 0]
            for nk, h in zip(nz, holm([r[nk][cmp_]["p_exact"] for nk in nz])): r[nk][cmp_]["p_holm"] = h
        # splits 0 to 4 of the submitted version, shown next to it and never pooled with it
        o = json.load(open(ROOT / "results" / ORIG6[tag]))["datasets"][f]["by_noise"]
        for nk in r:
            if nk in o: r[nk]["splits_0_to_4_means"] = {k: float(np.mean(o[nk]["fusion"][k]["mcc_seeds"])) for k in ("QVE3", "QWE3", "NWE3@0.05")}
        res[name] = r
    return res

def s2():
    res = {}
    for dev in DEVS:
        for tag, (f, name) in DS.items():
            p = SUP / f"fusion_6q_{dev}_{tag}.json"
            if not p.exists(): continue
            bn = json.load(open(p))["datasets"][f]["by_noise"]
            key = [k for k in bn if k.startswith("device")]
            if not key: continue
            c = cell(bn[key[0]]); c["noiseless"] = cell(bn["0.0"]) if "0.0" in bn else None
            c["spread_retention"] = {m: float(bn[key[0]]["branch"][m]["spread"] / bn["0.0"]["branch"][m]["spread"]) for m in bn[key[0]]["branch"]} if "0.0" in bn else None
            res.setdefault(dev, {})[name] = c
    return res

def s3():
    res = {}
    st = SUP / "structure_rs.json"; ef = SUP / "effective_params_rs.json"
    if st.exists():
        S = json.load(open(st)); E = json.load(open(ef)) if ef.exists() else {}
        for k, v in S.items():
            res.setdefault("structure", {})[k] = {**{x: v[x] for x in ("per_member_2q", "total_2q", "reduction_pct", "distinct_members", "member_md5")},
                "effective": sum(E.get(f"{k}__{t}", {}).get("effective", 0) for t in ("Z_1_full", "ZZ_2_full", "Pauli_1_full"))}
    for tag, (f, name) in DS.items():
        p = SUP / f"compression_matched_rs_{tag}.json"; q = ROOT / f"results/corrected/compression_matched_corrected_{tag}.json"
        if not p.exists(): continue
        bn = json.load(open(p))["datasets"][f]["by_noise"]; cb = json.load(open(q))["datasets"][f]["by_noise"]; r = {}
        for nk in bn:
            r[nk] = {}
            for rule in ("QVE3", "NWE3"):
                rs = {a: float(np.mean(bn[nk][a]["fusion"][rule]["mcc_seeds"])) for a in bn[nk] if a.startswith("rs_")}
                co = {a: float(np.mean(cb[nk][a]["fusion"][rule]["mcc_seeds"])) for a in cb[nk] if a.startswith("corr_lr5_")}
                r[nk][rule] = {"reward_seed_equals_training_seed": rs, "reward_seed_42": co,
                               "seed42_shared_member": float(np.mean(cb[nk]["corr_lr5_s42"]["fusion"][rule]["mcc_seeds"])),
                               "uncompressed": float(np.mean(cb[nk]["uncompressed"]["fusion"][rule]["mcc_seeds"])), "linear": float(np.mean(cb[nk]["linear"]["fusion"][rule]["mcc_seeds"]))}
        res.setdefault("downstream", {})[name] = r
    return res

def tex_s1(R):
    L = []
    for name, r in R.items():
        nks = sorted(r, key=float); L.append(f"\\multirow{{{len(nks)}}}{{*}}{{{name}}}")
        for nk in nks:
            c = r[nk]; v = c["NWE3_vs_QVE3"]
            ph = f"${v['p_holm']:.4f}$" if "p_holm" in v else "n/a"; pe = f"${v['p_exact']:.4f}$" if float(nk) > 0 else "n/a"
            L.append(f" & ${float(nk):g}$ & ${c['QVE3']['mean']:.3f} \\pm {c['QVE3']['sd']:.3f}$ & ${c['QWE3']['mean']:.3f} \\pm {c['QWE3']['sd']:.3f}$ & ${c['NWE3@0.05']['mean']:.3f} \\pm {c['NWE3@0.05']['sd']:.3f}$ & ${v['wins']}/{v['ties']}/{v['losses']}$ & {pe} & {ph} \\\\")
        L.append("\\midrule")
    body = "\n".join(L[:-1])
    (TEX / "table_s1_splits5to14.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Six-qubit coupled family on the ten confirmation splits (data-split seeds 5 to
14), run at a reviewer's request under the protocol of Table~\ref{tab:fusion6q} and
reported separately from splits 0 to 4. Mean and sample standard deviation of the Matthews
correlation; wins, ties and losses of the noise-aware rule against majority voting; exact
two-sided Wilcoxon $p$ with tied pairs dropped, and Holm correction across the six
non-zero levels of a dataset.}
\label{tab:s1splits}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Dataset & $p_1$ & QVE3 & QWE3 & NWE3 & W/T/L & $p$ & Holm \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""")

def tex_s2(R):
    L = []
    for dev, d in R.items():
        L.append(f"\\multirow{{{len(d)}}}{{*}}{{\\texttt{{{dev.replace('_', chr(92) + '_')}}}}}")
        for name, c in d.items():
            v = c["NWE3_vs_QVE3"]; b = c["branch_mean"]
            L.append(f" & {name} & ${c['QVE3']['mean']:.3f}$ & ${c['QWE3']['mean']:.3f}$ & ${c['NWE3@0.05']['mean']:.3f}$ & ${b['Z1']:.3f}$ & ${b['ZZ2']:.3f}$ & ${b['Pauli1']:.3f}$ & ${v['wins']}/{v['ties']}/{v['losses']}$ \\\\")
        L.append("\\midrule")
    (TEX / "table_s2_device.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Fusion rules at six qubits under device-derived noise models built from the
calibration snapshots of six seven-qubit processors, in simulation, over five splits.
Matthews correlation, mean; wins, ties and losses of the noise-aware rule against majority
voting. The kernel is measurement-free, so readout error and shot noise are excluded.}
\label{tab:s2device}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrrrrrrr@{}}
\toprule
Snapshot & Dataset & QVE3 & QWE3 & NWE3 & $Z$ & $ZZ$ & Pauli & W/T/L \\
\midrule
""" + "\n".join(L[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

def tex_s3(R):
    st = R.get("structure", {}); dn = R.get("downstream", {})
    corr = json.load(open(ROOT / "results/corrected/structure.json"))
    L = []
    rows = [("corr_lr5_s42", "42", corr["corr_lr5_s42"])] + [(k, k.split("_s")[-1], v) for k, v in sorted(st.items())]
    for name, seed, v in rows:
        pm = "/".join(str(x) for x in v["per_member_2q"]); same = "yes" if v["member_md5"][1] == "547e3a9f" and v["member_md5"][2] == "547e3a9f" else "no"
        eff = v.get("effective", 18)
        L.append(f"{seed} & ${pm}$ & ${v['total_2q']}$ & ${v['reduction_pct']:.1f}\\%$ & ${v['distinct_members']}$ & {same} & ${eff}/18$ \\\\")
    (TEX / "table_s3_structure.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Committees emitted at the confirmatory learning rate when the reward subsample
is drawn with the training seed instead of the fixed seed 42. Seed 42 is the run of
Table~\ref{tab:compstructure} whose reward seed already equals its training seed.
``Warm-up circuit'' marks committees whose two entangling members are byte-identical to
the linear-entanglement Pauli circuit that the supervised warm-up teaches.}
\label{tab:s3structure}
\begin{tabular}{@{}lrrrrlr@{}}
\toprule
Seed & Per-member $g_2$ & Total $g_2$ & Reduction & Distinct & Warm-up circuit & Effective \\
\midrule
""" + "\n".join(L) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    L = []
    for name, r in dn.items():
        nks = sorted(r, key=float); L.append(f"\\multirow{{{len(nks)}}}{{*}}{{{name}}}")
        for nk in nks:
            cells = []
            for rule in ("QVE3", "NWE3"):
                x = r[nk][rule]; rs = list(x["reward_seed_equals_training_seed"].values()); co = list(x["reward_seed_42"].values())
                cells += [f"${np.mean(rs):.3f}$", f"${np.mean(co):.3f}$", f"${x['linear']:.3f}$"]
            L.append(f" & ${float(nk):g}$ & " + " & ".join(cells) + " \\\\")
        L.append("\\midrule")
    (TEX / "table_s3_downstream.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Downstream Matthews correlation of the reward-seeded committees (mean over
policy seeds 43 to 46 and over five data splits) next to the shared-subsample committees
of Tables~\ref{tab:compdownstream} and~\ref{tab:compdownstream-nwe} (seeds 42 to 46)
and the hand-designed substitution, coupled family, six qubits.}
\label{tab:s3downstream}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
& & \multicolumn{3}{c}{QVE3, majority voting} & \multicolumn{3}{c}{NWE3, noise-aware} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Dataset & $p_1$ & seeded & shared & linear & seeded & shared & linear \\
\midrule
""" + "\n".join(L[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

if __name__ == "__main__":
    out = {"S1": s1(), "S2": s2(), "S3": s3()}
    atomic_write_json(SUP / "summary_supplement.json", out, indent=1)
    if out["S1"]: tex_s1(out["S1"])
    if out["S2"]: tex_s2(out["S2"])
    if out["S3"]: tex_s3(out["S3"])
    for name, r in out["S1"].items():
        print("S1", name)
        for nk in sorted(r, key=float):
            v = r[nk]["NWE3_vs_QVE3"]; w = r[nk]["NWE3_vs_QWE3"]
            print(f"   p1={nk:<6} QVE {r[nk]['QVE3']['mean']:.3f} QWE {r[nk]['QWE3']['mean']:.3f} NWE {r[nk]['NWE3@0.05']['mean']:.3f} | vsQVE {v['wins']}/{v['ties']}/{v['losses']} p={v['p_exact']:.4f} holm={v.get('p_holm', float('nan')):.4f} | vsQWE {w['wins']}/{w['ties']}/{w['losses']} p={w['p_exact']:.4f}")
    for dev, d in out["S2"].items():
        for name, c in d.items():
            v = c["NWE3_vs_QVE3"]; print(f"S2 {dev:<16}{name:<10} QVE {c['QVE3']['mean']:.3f} QWE {c['QWE3']['mean']:.3f} NWE {c['NWE3@0.05']['mean']:.3f} Z {c['branch_mean']['Z1']:.3f} ZZ {c['branch_mean']['ZZ2']:.3f} P {c['branch_mean']['Pauli1']:.3f} {v['wins']}/{v['ties']}/{v['losses']} retention {c['spread_retention']}")
    if out["S3"]: print("S3", json.dumps(out["S3"].get("structure", {}), indent=0)[:1500])
    print("[written]", SUP / "summary_supplement.json")
