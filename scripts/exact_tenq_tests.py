#!/usr/bin/env python
"""Exact two-sided Wilcoxon signed-rank tests (tied pairs dropped) for the ten-qubit,
p2 = 0.02 comparison on UNSW-NB15, sequential (splits 0 to 14) and confirmation (5 to 14),
with Holm across the two comparisons. The merged file stores a zero-split variant computed
at run time; the article reports the values written here."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from taqcc.io import atomic_write_json
EPS = 1e-9
def exact(d):
    d = np.asarray(d, float); nz = d[np.abs(d) > EPS]
    return float(wilcoxon(nz, zero_method="wilcox", alternative="two-sided", method="exact").pvalue) if len(nz) else 1.0
def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = [0.0] * m; run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj
out = {}
for name, f, sl in (("sequential_n15", "results/fusion_10q_merged15.json", slice(0, 15)), ("confirmation_n10", "results/fusion_10q_merged15.json", slice(5, 15))):
    fu = json.load(open(ROOT / f))["fusion"]; n = np.array(fu["NWE3@0.05"]["mcc_seeds"])[sl]
    res = {}
    for other in ("QVE3", "QWE3"):
        d = n - np.array(fu[other]["mcc_seeds"])[sl]
        res[f"NWE3_vs_{other}"] = {"n": int(len(d)), "mean_delta": float(d.mean()), "wins": int((d > EPS).sum()), "ties": int((np.abs(d) <= EPS).sum()), "losses": int((d < -EPS).sum()), "p_exact_ties_dropped": exact(d)}
    ps = holm([res[k]["p_exact_ties_dropped"] for k in res])
    for k, h in zip(res, ps): res[k]["p_holm"] = float(h)
    out[name] = res
out["means_n15"] = {r: float(np.mean(json.load(open(ROOT / "results/fusion_10q_merged15.json"))["fusion"][r]["mcc_seeds"])) for r in ("QVE3", "QWE3", "NWE3@0.05")}
atomic_write_json(ROOT / "results/corrected/tenq15_exact_tests.json", out, indent=1)
print(json.dumps(out, indent=1))
