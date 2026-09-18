#!/usr/bin/env python
"""Check every headline number in the SN Computer Science article against results/.

The article's Code availability statement points reviewers here, so this script exists
to let a reviewer confirm the tables without rerunning anything. Values below are typed
from the printed tables; each is compared against the JSON that produced it. A mismatch
means the paper and the repository disagree, which is exactly what a reader would want
to find out.

Reads only `results/*.json`. No GPU, no simulation, runs in about a second.

  python scripts/verify_paper_numbers.py
  python scripts/verify_paper_numbers.py --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
TOL = 0.0006  # printed to three decimals

FUSION6Q = {  # Table 4, coupled family p2 = 10 p1, six qubits
    "IoTID20": ("fusion_v2_IoT_Orig.json", "IoT_Original_Distribution.csv", {
        #        QVE3   QWE3   NWE3  |   Z      ZZ    Pauli
        "0.0":  (0.675, 0.675, 0.675, 0.548, 0.662, 0.675),
        "0.01": (0.000, 0.000, 0.546, 0.546, 0.000, 0.000),
        "0.03": (0.000, 0.000, 0.550, 0.550, 0.000, 0.000),
        "0.05": (0.000, 0.000, 0.535, 0.535, 0.000, 0.000),
        "0.1":  (0.000, 0.000, 0.514, 0.514, 0.000, 0.000)}),
    "UNSW-NB15": ("fusion_v2_UNSW_NB1.json", "UNSW_NB15.csv", {
        "0.0":  (0.667, 0.667, 0.667, 0.649, 0.659, 0.670),
        "0.01": (0.000, 0.651, 0.651, 0.651, 0.000, 0.000),
        "0.03": (0.000, 0.632, 0.632, 0.632, 0.000, 0.000),
        "0.05": (0.000, 0.629, 0.629, 0.629, 0.000, 0.000),
        "0.1":  (0.000, 0.615, 0.615, 0.615, 0.000, 0.000)}),
    "Bot-IoT": ("fusion_v2_UNSW_201.json", "UNSW_2018_IoT_Botnet_Final_10_Best.csv", {
        "0.0":  (0.846, 0.846, 0.846, 0.839, 0.834, 0.842),
        "0.01": (0.000, 0.000, 0.839, 0.839, 0.000, 0.000),
        "0.03": (0.000, 0.000, 0.838, 0.838, 0.000, 0.000),
        "0.05": (0.000, 0.000, 0.833, 0.833, 0.000, 0.000),
        "0.1":  (0.000, 0.000, 0.820, 0.820, 0.000, 0.000)}),
}

TAU = {  # Table 5, NWE3 against the gate threshold, IoTID20
    "0.0":  (0.675, 0.675, 0.675, 0.675, 0.675),
    "0.01": (0.428, 0.546, 0.546, 0.546, 0.546),
    "0.1":  (0.514, 0.514, 0.514, 0.514, 0.514),
}
TAUS = ["0.01", "0.02", "0.05", "0.1", "0.2"]

REALISTIC = {  # Table 6, p1 = 5e-4, UNSW-NB15
    8:  ("fusion_realistic_8q.json", {
        "0.0:0.0":    (0.639, 0.639, 0.639, 0.638, 0.616, 0.652),
        "0.0005:0.01": (0.600, 0.600, 0.600, 0.639, 0.553, 0.590),
        "0.0005:0.02": (0.641, 0.641, 0.641, 0.639, 0.023, 0.545)}),
    10: ("fusion_realistic_10q.json", {
        "0.0:0.0":     (0.678, 0.678, 0.678, 0.651, 0.640, 0.670),
        "0.0005:0.01": (0.656, 0.656, 0.656, 0.649, 0.166, 0.561),
        "0.0005:0.02": (0.484, 0.571, 0.649, 0.649, 0.000, 0.448)}),
}

STRUCTURE = {  # Table 2, total two-qubit gates / distinct members / effective params
    "grpo_fix_lr5":  (20, 2, 18),
    "grpo_fix_lr75": (0,  1, 18),
    "grpo_fix_lr10": (90, 3, 18),
    "rep_lr5_s43":   (7,  3,  8),   # the arm that games the structural criterion
    "rep_lr75_s43":  (90, 3, 18),
    "rep_lr10_s43":  (90, 3, 18),
    "sft_warmup":    (30, 3, 18),
}

AGREEMENT = {  # Table 8, Yule's Q per pair, then majority / best / oracle
    "IoT_Original_Distribution.csv":         (0.850, 0.935, 0.998, 0.675, 0.682, 0.802),
    "UNSW_NB15.csv":                         (0.921, 0.944, 0.993, 0.667, 0.685, 0.773),
    "UNSW_2018_IoT_Botnet_Final_10_Best.csv": (0.951, 0.976, 0.999, 0.846, 0.878, 0.928),
}

CLASSICAL = {  # Table 9, mean MCC over five seeds
    "RandomForest":       (0.697, 0.710, 0.853),
    "SVM_RBF":            (0.515, 0.650, 0.827),
    "LogisticRegression": (0.462, 0.557, 0.838),
}
CLASSICAL_DS = ["IoT_Original_Distribution.csv", "UNSW_NB15.csv",
                "UNSW_2018_IoT_Botnet_Final_10_Best.csv"]

RULES = ["QVE3", "QWE3", "NWE3@0.05"]
BRANCHES = ["Z1", "ZZ2", "Pauli1"]


class Check:
    def __init__(self, verbose):
        self.verbose, self.fail, self.n, self.skipped = verbose, [], 0, []

    def eq(self, label, got, want, tol=TOL):
        self.n += 1
        ok = got is not None and abs(float(got) - float(want)) <= tol
        if not ok:
            self.fail.append(f"{label}: paper {want}, results {got}")
        elif self.verbose:
            print(f"  ok  {label}  {want}")
        return ok

    def load(self, name):
        p = RESULTS / name
        if not p.exists():
            self.skipped.append(name)
            return None
        return json.loads(p.read_text())


# ---------------------------------------------------------------------------------------
# Revision: the corrected campaign tables are generated from results/corrected/ by
# make_revision_tables.py, so the check here is that the generated LaTeX still matches
# the JSON it was built from (guards against a stale tables/ copy in the manuscript).
def check_corrected(c):
    import re
    corr = RESULTS / "corrected"
    st = c.load("corrected/structure.json"); eff = c.load("corrected/effective_params.json")
    tex = corr / "tex" / "table2_corrected.tex"
    if st and eff and tex.exists():
        # corrected rows start with "& seed" (the multirow label is on its own line);
        # reference rows start with a name and are not policies
        rows = re.findall(r"^ & (\d+) & \$(\d+)/(\d+)/(\d+)\$ & \$(\d+)\$ & \$([\d.]+)\\%\$ & \$(\d+)\$ & \$(\d+)/18\$", tex.read_text(), re.M)
        lrs = ["lr5"] * 5 + ["lr75"] * 5 + ["lr10"] * 5
        for (seed, z, zz, pa, tot, red, dist, e), lr in zip(rows, lrs):
            n = f"corr_{lr}_s{seed}"; v = st[n]
            c.eq(f"T2c {n} per-member", v["per_member_2q"] == [int(z), int(zz), int(pa)], True, 0)
            c.eq(f"T2c {n} total", v["total_2q"], int(tot), 0)
            c.eq(f"T2c {n} distinct", v["distinct_members"], int(dist), 0)
            c.eq(f"T2c {n} effective", sum(eff[f"{n}__{t}"]["effective"] for t in ("Z_1_full", "ZZ_2_full", "Pauli_1_full")), int(e), 0)
        print(f"Table 2 (corrected)  {len(rows)} policies checked against structure.json and effective_params.json")
    for tag, dsf in (("IoT", "IoT_Original_Distribution.csv"), ("UNSW", "UNSW_NB15.csv"), ("Bot", "UNSW_2018_IoT_Botnet_Final_10_Best.csv")):
        b = c.load(f"corrected/compression_matched_corrected_{tag}.json")
        if not b: continue
        bn = b["datasets"][dsf]["by_noise"]
        for nk in bn:
            for arm in ("uncompressed", "linear"):
                for r in ("QVE3", "NWE3"):
                    c.eq(f"T3c {tag} {arm} {nk} {r} n_splits", len(bn[nk][arm]["fusion"][r]["mcc_seeds"]), 5, 0)
    print("Table 3 (corrected)  split counts checked")
    # realistic family and tau sweep at 8 and 10 qubits: every printed cell against its JSON
    files = {(8, "UNSW-NB15"): "fusion_realistic_8q.json", (10, "UNSW-NB15"): "fusion_realistic_10q.json",
             (8, "IoTID20"): "corrected/fusion_realistic_8q_IoT.json", (10, "IoTID20"): "corrected/fusion_realistic_10q_IoT.json",
             (8, "Bot-IoT"): "corrected/fusion_realistic_8q_Bot.json", (10, "Bot-IoT"): "corrected/fusion_realistic_10q_Bot.json"}
    num = re.compile(r"(\d\.\d{3})")
    for texname, keys in (("table_realistic_all.tex", ["QVE3", "QWE3", "NWE3@0.05", "b:Z1", "b:ZZ2", "b:Pauli1"]),
                          ("table_tau_widths.tex", ["NWE3@0.01", "NWE3@0.02", "NWE3@0.05", "NWE3@0.1", "NWE3@0.2", "QVE3", "b:Z1"])):
        tex = corr / "tex" / texname
        if not tex.exists():
            c.skipped.append(texname); continue
        cur, n = None, 0
        for line in tex.read_text().splitlines():
            m = re.match(r"\\multirow\{3\}\{\*\}\{(\d+), ([^}]+)\}", line)
            if m:
                d = c.load(files[(int(m.group(1)), m.group(2))])
                cur = (m.group(1), m.group(2), list(d["datasets"].values())[0]["by_noise"]) if d else None
                continue
            if cur is None or not line.startswith(" & $"): continue
            p2 = re.match(r" & \$([\d.]+)\$", line).group(1)
            nk = next(k for k in cur[2] if float(k.split(":")[-1]) == float(p2))
            vals = num.findall(line.split("&", 2)[2])
            for key, v in zip(keys, vals):
                cell = cur[2][nk]
                got = cell["branch"][key[2:]]["mcc"] if key.startswith("b:") else cell["fusion"][key]["mcc_mean"]
                c.eq(f"{texname} {cur[0]}q {cur[1]} p2={p2} {key}", got, float(v)); n += 1
        print(f"{texname}  {n} cells checked")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    c = Check(args.verbose)

    print("Table 2  committee structure and effective parameters")
    struct = c.load("replicates_structure.json")
    eff = c.load("effective_params.json")
    if struct and eff:
        for model, (g2, distinct, effective) in STRUCTURE.items():
            if model in struct:
                c.eq(f"T2 {model} total 2q", struct[model]["total_2q"], g2, 0)
                c.eq(f"T2 {model} distinct", struct[model]["distinct_members"], distinct, 0)
            got = sum(v["effective"] for k, v in eff.items() if k.startswith(model + "__"))
            if got:
                c.eq(f"T2 {model} effective", got, effective, 0)

    print("Tables 4 and 5  fusion under the coupled family, six qubits")
    for ds, (fname, dskey, rows) in FUSION6Q.items():
        blob = c.load(fname)
        if not blob:
            continue
        bn = blob["datasets"][dskey]["by_noise"]
        for p1, vals in rows.items():
            got = [bn[p1]["fusion"][r]["mcc_mean"] for r in RULES]
            got += [bn[p1]["branch"][b]["mcc"] for b in BRANCHES]
            for name, g, w in zip(RULES + BRANCHES, got, vals):
                c.eq(f"T4 {ds} p1={p1} {name}", g, w)
        if ds == "IoTID20":
            for p1, vals in TAU.items():
                for tau, w in zip(TAUS, vals):
                    c.eq(f"T5 {ds} p1={p1} tau={tau}",
                         bn[p1]["fusion"][f"NWE3@{tau}"]["mcc_mean"], w)

    print("Table 6  hardware-realistic noise, UNSW-NB15")
    for width, (fname, rows) in REALISTIC.items():
        blob = c.load(fname)
        if not blob:
            continue
        bn = blob["datasets"]["UNSW_NB15.csv"]["by_noise"]
        for nk, vals in rows.items():
            got = [bn[nk]["fusion"][r]["mcc_mean"] for r in RULES]
            got += [bn[nk]["branch"][b]["mcc"] for b in BRANCHES]
            for name, g, w in zip(RULES + BRANCHES, got, vals):
                c.eq(f"T6 {width}q {nk} {name}", g, w)

    print("Table 7  fifteen paired splits at ten qubits")
    m = c.load("fusion_10q_merged15.json")
    if m:
        c.eq("T7 n seeds", m["n"], 15, 0)
        c.eq("T7 QVE3 mean", m["fusion"]["QVE3"]["mcc_mean"], 0.482)
        c.eq("T7 QWE3 mean", m["fusion"]["QWE3"]["mcc_mean"], 0.574)
        c.eq("T7 NWE3 mean", m["fusion"]["NWE3@0.05"]["mcc_mean"], 0.664)
        t = m["tests"]["NWE3@0.05_vs_QVE3"]
        c.eq("T7 vs QVE3 wins", t["wins"], 12, 0)
        c.eq("T7 vs QVE3 ties", t["ties"], 3, 0)
        c.eq("T7 vs QVE3 losses", t["losses"], 0, 0)
        c.eq("T7 vs QVE3 p_holm", t["p_holm"], 0.0024, 0.00005)

    print("Table 8  member agreement and the oracle ceiling")
    a = c.load("member_agreement.json")
    if a:
        for dskey, vals in AGREEMENT.items():
            r = a["datasets"][dskey]
            pairs = ["Z-ZZ", "Z-Pauli", "ZZ-Pauli"]
            for pk, w in zip(pairs, vals[:3]):
                c.eq(f"T8 {dskey[:12]} Q {pk}", r["pairs"][pk]["q"], w)
            for key, w, nm in zip(["majority_mcc", "best_member_mcc", "oracle_mcc"],
                                  vals[3:], ["majority", "best", "oracle"]):
                c.eq(f"T8 {dskey[:12]} {nm}", r[key], w)

    print("Table 9  classical reference point")
    cb = c.load("classical_baseline_200.json")
    if cb:
        for model, vals in CLASSICAL.items():
            for dskey, w in zip(CLASSICAL_DS, vals):
                c.eq(f"T9 {model} {dskey[:12]}", cb["datasets"][dskey][model]["mcc"][0], w)

    print("Corrected campaign (revision)")
    check_corrected(c)

    print()
    if c.skipped:
        print(f"[skipped] {len(c.skipped)} result file(s) absent: {', '.join(c.skipped)}")
    if c.fail:
        print(f"[FAIL] {len(c.fail)} of {c.n} checks disagree with the article:")
        for f in c.fail:
            print("   " + f)
        return 1
    print(f"[OK] all {c.n} published values reproduce from results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
