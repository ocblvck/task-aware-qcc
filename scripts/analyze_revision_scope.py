#!/usr/bin/env python
"""Two scope checks for the revision, recomputed over everything the revised article reports.

1. Floor-referenced gate: admit a member iff its off-diagonal spread exceeds k * 2^-n, and
   count agreement with the article's relative rule (spread >= 0.05 * noiseless spread) over
   every stored member, noise level, dataset and width, including the revision runs.
2. Cost-weight sensitivity of the compression gain for the fifteen corrected committees:
   gamma = 1 - cost(comp)/cost(orig), cost = depth + w * n_2q, for w in 0, 1, 2, 5, 10.

Reads results/ only (plus the emitted circuits). No GPU.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from taqcc.io import atomic_write_json
R = ROOT / "results"; OUT = R / "corrected"

FILES = {6: ["fusion_v2_IoT_Orig.json", "fusion_v2_UNSW_NB1.json", "fusion_v2_UNSW_201.json",
             "corrected/c_sweep_IoT.json", "corrected/c_sweep_UNSW.json", "corrected/c_sweep_Bot.json"],
         8: ["fusion_realistic_8q.json", "corrected/fusion_realistic_8q_IoT.json", "corrected/fusion_realistic_8q_Bot.json"],
         10: ["fusion_realistic_10q.json", "corrected/fusion_realistic_10q_IoT.json", "corrected/fusion_realistic_10q_Bot.json"]}
KS = [1, 2, 5, 10, 20, 50]

def floor_gate():
    seen, cases = set(), []
    for n, files in FILES.items():
        for f in files:
            d = json.load(open(R / f))
            for ds, v in d["datasets"].items():
                bn = v["by_noise"]; clean = bn["0.0"] if "0.0" in bn else bn["0.0:0.0"]
                for nk, cell in bn.items():
                    for m, b in cell["branch"].items():
                        if m not in ("Z1", "ZZ2", "Pauli1"): continue   # committee members only, not ablation branches
                        key = (n, ds, float(nk.split(":")[-1]) if ":" in nk else float(nk), ":" in nk, m)
                        if key in seen: continue          # the C sweep repeats two levels of the main grid
                        seen.add(key)
                        s, s0 = b["spread"], clean["branch"][m]["spread"]
                        cases.append({"width": n, "dataset": ds, "noise": nk, "member": m, "spread": s,
                                      "noiseless_spread": s0, "relative_admits": bool(s >= 0.05 * s0)})
    out = {"note": __doc__.split("2.")[0].strip(), "n_cases": len(cases), "agreement": {}, "disagreements": {}}
    for k in KS:
        dis = [c for c in cases if bool(c["spread"] > k * 2.0 ** -c["width"]) != c["relative_admits"]]
        out["agreement"][str(k)] = {"match": len(cases) - len(dis), "total": len(cases)}
        out["disagreements"][str(k)] = dis
    s0s = [c["noiseless_spread"] for c in cases]
    out["noiseless_spread_range"] = [min(s0s), max(s0s)]
    return out

def cost_weights():
    from qiskit import qasm3
    from taqcc.feature_maps import circuit_metrics
    out = {"weights": [0, 1, 2, 5, 10], "arms": {}}
    names = sorted({p.name.split("__")[0] for p in (OUT / "circuits").glob("*.comp.qasm")})
    for name in names:
        dep, g2 = {"orig": [], "comp": []}, {"orig": [], "comp": []}
        for t in ("Z_1_full", "ZZ_2_full", "Pauli_1_full"):
            for kind in ("orig", "comp"):
                m = circuit_metrics(qasm3.loads((OUT / "circuits" / f"{name}__{t}.{kind}.qasm").read_text()))
                dep[kind].append(m["depth"]); g2[kind].append(m["two_qubit"] if "two_qubit" in m else m["n_2q"])
        gam = {}
        for w in out["weights"]:
            co = sum(dep["orig"]) + w * sum(g2["orig"]); cc = sum(dep["comp"]) + w * sum(g2["comp"])
            gam[str(w)] = round(1 - cc / co, 4)
        out["arms"][name] = {"depth": dep["comp"], "two_qubit": g2["comp"], "gamma_by_weight": gam}
    order = {str(w): sorted(out["arms"], key=lambda a: (-out["arms"][a]["gamma_by_weight"][str(w)], a)) for w in out["weights"]}
    out["ordering_by_weight"] = order
    from itertools import combinations
    flips = {}
    for w in out["weights"]:
        if w == 2: continue
        n = 0
        for a, b in combinations(out["arms"], 2):
            d2 = out["arms"][a]["gamma_by_weight"]["2"] - out["arms"][b]["gamma_by_weight"]["2"]
            dw = out["arms"][a]["gamma_by_weight"][str(w)] - out["arms"][b]["gamma_by_weight"][str(w)]
            if d2 * dw < 0: n += 1
        flips[str(w)] = n
    out["pairwise_order_reversals_vs_weight_2"] = flips
    return out

if __name__ == "__main__":
    fg = floor_gate(); atomic_write_json(OUT / "absolute_floor_gate_all.json", fg, indent=1)
    print("floor gate:", fg["n_cases"], "cases", fg["agreement"], "s0 range", fg["noiseless_spread_range"])
    for c in fg["disagreements"]["2"]: print("  k=2 disagree:", c["width"], c["dataset"][:8], c["noise"], c["member"], round(c["spread"], 5), c["relative_admits"])
    cw = cost_weights(); atomic_write_json(OUT / "cost_weight_sensitivity_corrected.json", cw, indent=1)
    print("cost weights: reversals vs w=2:", cw["pairwise_order_reversals_vs_weight_2"])
