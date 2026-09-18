#!/usr/bin/env python
"""Emit LaTeX tables for the revised manuscript straight from results/corrected/, so
no number in the revised Section 8 is typed by hand. Writes results/corrected/tex/*.tex
and prints the headline numbers the prose quotes."""
import json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/"results/corrected"; TEX = OUT/"tex"; TEX.mkdir(exist_ok=True)
LRS=[("lr5","$5\\times10^{-6}$"),("lr75","$7.5\\times10^{-6}$"),("lr10","$1\\times10^{-5}$")]; SEEDS=[42,43,44,45,46]
DS={"IoT":("IoT_Original_Distribution.csv","IoTID20"),"UNSW":("UNSW_NB15.csv","UNSW-NB15"),"Bot":("UNSW_2018_IoT_Botnet_Final_10_Best.csv","Bot-IoT")}
st=json.load(open(OUT/"structure.json")); eff=json.load(open(OUT/"effective_params.json"))
pre=json.load(open(ROOT/"results/replicates_structure.json")); preeff=json.load(open(ROOT/"results/effective_params.json"))
def effsum(prefix, src): return sum(src.get(f"{prefix}__{t}",{}).get("effective",0) for t in ("Z_1_full","ZZ_2_full","Pauli_1_full"))
def w(name, s): (TEX/name).write_text(s); print("[tex]", TEX/name)

# ---------------- Table 2 (corrected): structure per policy
rows=[]
for lr,lrtex in LRS:
    for s in SEEDS:
        n=f"corr_{lr}_s{s}"; v=st[n]; e=effsum(n,eff)
        rows.append(f" & {s} & ${'/'.join(map(str,v['per_member_2q']))}$ & ${v['total_2q']}$ & ${v['reduction_pct']:.1f}\\%$ & ${v['distinct_members']}$ & ${e}/18$ \\\\")
    rows.insert(len(rows)-5, f"\\multirow{{5}}{{*}}{{GRPO {lrtex}}}")
body=[]
for lr,lrtex in LRS:
    body.append(f"\\multirow{{5}}{{*}}{{GRPO {lrtex}}}")
    for s in SEEDS:
        n=f"corr_{lr}_s{s}"; v=st[n]; e=effsum(n,eff)
        body.append(f" & {s} & ${'/'.join(map(str,v['per_member_2q']))}$ & ${v['total_2q']}$ & ${v['reduction_pct']:.1f}\\%$ & ${v['distinct_members']}$ & ${e}/18$ \\\\")
    body.append("\\midrule")
body=body[:-1]
t2 = r"""\begin{table}[t]
\centering
\caption{Committee structure after compression at six qubits under the corrected
objective, every learning rate at five training seeds. Two-qubit gate counts per member
in the order $Z$, $ZZ$, Pauli; distinct counts how many of the three members are still
different circuits; effective counts parameters passing the perturbation test of
\eqref{eq:effective}, which every policy was trained against. Reference arms are those of
Table~\ref{tab:compstructure-pre}.}
\label{tab:compstructure}
\begin{tabular}{@{}llcccrc@{}}
\toprule
Arm & Seed & Per-member $g_2$ & Total $g_2$ & Reduction & Distinct & Effective \\
\midrule
Uncompressed & n/a & $0/60/30$ & $90$ & $0\%$ & $3$ & $18/18$ \\
Hand-designed linear & n/a & $0/20/10$ & $30$ & $66.7\%$ & $3$ & $18/18$ \\
Supervised warm-up & 42 & $0/20/10$ & $30$ & $66.7\%$ & $3$ & $18/18$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
w("table2_corrected.tex", t2)

# ---------------- Table 2-pre (appendix): submitted policies with the Eq. 9 audit
pb=[]
for lr,lrtex in LRS:
    for s,pn in ((42,f"grpo_fix_{lr}"),(43,f"rep_{lr}_s43")):
        v=pre[pn]; e=effsum(pn,preeff)
        ecell = ("$\\mathbf{%d/18}$" % e) if e < 18 else ("$%d/18$" % e)
        g2 = "/".join(map(str, v["per_member_2q"]))
        pb.append("GRPO %s & %d & $%s$ & $%d$ & $%.1f\\%%$ & $%d$ & %s \\\\" % (lrtex, s, g2, v["total_2q"], v["reduction_pct"], v["distinct_members"], ecell))
w("table2_pre.tex", r"""\begin{table}[t]
\centering
\caption{The policies of the submitted version, trained against the structural criterion
alone with accuracy retention on a 16/8 subsample, audited after training with
\eqref{eq:effective}. The seed-43 policy at the lowest rate is the one that satisfied the
structural criterion while leaving ten of eighteen parameters inert.}
\label{tab:compstructure-pre}
\begin{tabular}{@{}llcccrc@{}}
\toprule
Arm & Seed & Per-member $g_2$ & Total $g_2$ & Reduction & Distinct & Effective \\
\midrule
""" + "\n".join(pb) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- Table 3 (corrected): downstream by lr, one table per fusion rule
def load(tag,kind="corrected"): return json.load(open(OUT/f"compression_matched_{kind}_{tag}.json"))
noises=["0.0","0.01","0.03","0.05","0.1"]
def cell(bn,arm,nk,rule): return float(np.mean(bn[nk][arm]["fusion"][rule]["mcc_seeds"]))
for rule,label,suffix,cap in (("QVE3","majority voting","", "under majority voting"),("NWE3","the noise-aware rule","-nwe","under the noise-aware rule")):
    lines=[]
    for tag,(dsf,dsn) in DS.items():
        bn=load(tag)["datasets"][dsf]["by_noise"]
        lines.append(f"\\multirow{{5}}{{*}}{{{dsn}}}")
        for arm,lab in (("uncompressed","Uncompressed"),("linear","Hand-designed linear")):
            lines.append(f" & {lab} & "+" & ".join(f"${cell(bn,arm,nk,rule):.3f}$" for nk in noises)+" \\\\")
        for lr,lrtex in LRS:
            arms=[f"corr_{lr}_s{s}" for s in SEEDS]; q=[]
            for nk in noises:
                m=[cell(bn,a,nk,rule) for a in arms]; q.append(f"${np.mean(m):.3f}$ ({np.std(m,ddof=1):.2f})")
            lines.append(f" & GRPO {lrtex} & "+" & ".join(q)+" \\\\")
        lines.append("\\midrule")
    lines=lines[:-1]
    w(f"table3_corrected{suffix}.tex", r"""\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\caption{Downstream Matthews correlation of the corrected policies at six qubits under
the coupled family, matched protocol (200/400, five data splits), """ + cap + r""".
Learned rows give the mean over five policy seeds with the standard deviation across
seeds in parentheses; each seed's value is itself a mean over the five data splits.
Per-seed values are in the repository.}
\label{tab:compdownstream""" + suffix + r"""}
\begin{tabular}{@{}llrrrrr@{}}
\toprule
Dataset & Arm & $p_1 = 0$ & $0.01$ & $0.03$ & $0.05$ & $0.1$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- headline numbers for the prose
print("\n=== headline numbers ===")
for lr,_ in LRS:
    vs=[st[f"corr_{lr}_s{s}"] for s in SEEDS]; es=[effsum(f"corr_{lr}_s{s}",eff) for s in SEEDS]
    print(f"{lr}: compressed {sum(v['total_2q']<90 for v in vs)}/5, g2 range {min(v['total_2q'] for v in vs)}-{max(v['total_2q'] for v in vs)}, mean reduction {np.mean([v['reduction_pct'] for v in vs]):.1f}%, 3 distinct in {sum(v['distinct_members']==3 for v in vs)}, ZZ=Pauli in {sum(v['member_md5'][1]==v['member_md5'][2] for v in vs)}, effective 18/18 in {sum(e==18 for e in es)}")
print("ZZ=Pauli overall:", sum(st[f'corr_{lr}_s{s}']['member_md5'][1]==st[f'corr_{lr}_s{s}']['member_md5'][2] for lr,_ in LRS for s in SEEDS), "of 15")
for tag,(dsf,dsn) in DS.items():
    bn=load(tag)["datasets"][dsf]["by_noise"]; arms=[f"corr_lr5_s{s}" for s in SEEDS]
    for nk in ("0.0","0.01","0.03"):
        d=[cell(bn,a,nk,"QVE3")-cell(bn,"linear",nk,"QVE3") for a in arms]; dn=[cell(bn,a,nk,"NWE3")-cell(bn,"linear",nk,"NWE3") for a in arms]
        print(f"{dsn} p1={nk}: lr5 minus linear QVE3 {[round(x,3) for x in d]}  NWE3 {[round(x,3) for x in dn]}")

# ---------------- Table: SVM C sweep (E4)
cs=[]
for tag,(dsf,dsn) in DS.items():
    bn=json.load(open(OUT/f"c_sweep_{tag}.json"))["datasets"][dsf]["by_noise"]
    cs.append(f"\\multirow{{8}}{{*}}{{{dsn}}}")
    for nk in ("0.005","0.01"):
        byc=bn[nk]["by_C"]
        for lab,name in (("ZZ2","$ZZ$"),("Pauli1","Pauli")):
            cs.append(f" & ${nk}$ & {name} & "+" & ".join(f"${byc[c]['branch'][lab]['mcc_mean']:.3f}$" for c in ("0.1","1.0","10.0","100.0"))+" \\\\")
        for r in ("QVE3","NWE3"):
            cs.append(f" & ${nk}$ & {r} & "+" & ".join(f"${byc[c]['fusion'][r]['mcc_mean']:.3f}$" for c in ("0.1","1.0","10.0","100.0"))+" \\\\")
    cs.append("\\midrule")
cs=cs[:-1]
w("table_csweep.tex", r"""\begin{table}[t]
\centering
\caption{Sensitivity to the support-vector regularization constant, six qubits, coupled
family, on the same Gram matrices. Matthews correlation, mean over five splits. The
article's tables use $C = 1$, fixed in advance. Off-diagonal spreads at $p_1 = 0.01$ are
about $3\times10^{-5}$ for $ZZ$ and $10^{-3}$ for Pauli.}
\label{tab:csweep}
\begin{tabular}{@{}lllrrrr@{}}
\toprule
Dataset & $p_1$ & Member or rule & $C = 0.1$ & $1$ & $10$ & $100$ \\
\midrule
""" + "\n".join(cs) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- Table: realistic family at 8 (and 10 when present) qubits, all datasets
rl=[]
for wq in (8,10):
    for tag,(dsf,dsn) in DS.items():
        f = ROOT/f"results/fusion_realistic_{wq}q.json" if tag=="UNSW" else OUT/f"fusion_realistic_{wq}q_{tag}.json"
        if not f.exists(): continue
        bn=json.load(open(f))["datasets"][dsf]["by_noise"]
        if len(bn)<3: continue
        rl.append(f"\\multirow{{3}}{{*}}{{{wq}, {dsn}}}")
        for nk,d in bn.items():
            fu=d["fusion"]; br=d["branch"]
            nwe=fu["NWE3@0.05"]["mcc_mean"]; qve=fu["QVE3"]["mcc_mean"]
            nwecell = f"$\\mathbf{{{nwe:.3f}}}$" if nwe>qve+5e-4 else (f"$\\underline{{{nwe:.3f}}}$" if nwe<qve-5e-4 else f"${nwe:.3f}$")
            rl.append(f" & ${nk.split(':')[-1]}$ & ${qve:.3f}$ & ${fu['QWE3']['mcc_mean']:.3f}$ & {nwecell} & ${br['Z1']['mcc']:.3f}$ & ${br['ZZ2']['mcc']:.3f}$ & ${br['Pauli1']['mcc']:.3f}$ \\\\")
        rl.append("\\midrule")
rl=rl[:-1]
w("table_realistic_all.tex", r"""\begin{table}[t]
\centering
\caption{Fusion rules at hardware-realistic error rates, $p_1 = 5\times10^{-4}$, on all
three datasets, matched protocol over five splits. The noise-aware value is bold where it
exceeds majority voting and underlined where it falls below it.}
\label{tab:fusionrealistic}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Width, dataset & $p_2$ & QVE3 & QWE3 & NWE3 & $Z$ & $ZZ$ & Pauli \\
\midrule
""" + "\n".join(rl) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- Table: gate threshold sweep at 8 and 10 qubits, all datasets
TAUS=["0.01","0.02","0.05","0.1","0.2"]; tl=[]
for wq in (8,10):
    for tag,(dsf,dsn) in DS.items():
        f = ROOT/f"results/fusion_realistic_{wq}q.json" if tag=="UNSW" else OUT/f"fusion_realistic_{wq}q_{tag}.json"
        if not f.exists(): continue
        bn=json.load(open(f))["datasets"][dsf]["by_noise"]
        if len(bn)<3: continue
        tl.append(f"\\multirow{{3}}{{*}}{{{wq}, {dsn}}}")
        for nk,d in bn.items():
            fu=d["fusion"]; ref=fu["NWE3@0.05"]["mcc_mean"]; cells=[]
            for t in TAUS:
                v=fu[f"NWE3@{t}"]["mcc_mean"]
                cells.append(f"$\\mathbf{{{v:.3f}}}$" if abs(v-ref)>5e-4 else f"${v:.3f}$")
            tl.append(f" & ${nk.split(':')[-1]}$ & " + " & ".join(cells) + f" & ${fu['QVE3']['mcc_mean']:.3f}$ & ${d['branch']['Z1']['mcc']:.3f}$ \\\\")
        tl.append("\\midrule")
tl=tl[:-1]
w("table_tau_widths.tex", r"""\begin{table}[t]
\centering
\caption{Sensitivity of the noise-aware rule to $\tau$ at eight and ten qubits on all
three datasets, hardware-realistic family, Matthews correlation over five splits. Cells
that differ from the $\tau = 0.05$ column are in bold. Majority voting and the shallow
member alone are given for reference.}
\label{tab:tauwidth}
\footnotesize
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
Width, dataset & $p_2$ & $\tau = 0.01$ & $0.02$ & $0.05$ & $0.1$ & $0.2$ & QVE3 & $Z$ alone \\
\midrule
""" + "\n".join(tl) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
