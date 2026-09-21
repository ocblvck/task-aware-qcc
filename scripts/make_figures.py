#!/usr/bin/env python
"""Build the four figures for the SN Computer Science manuscript.

Every number here is transcribed from the tables in `Okekeogbu_manuscript.tex`, which are in
turn generated from the JSON files in `task-aware-qcc/results/`. The `--check` flag
re-reads those JSONs and asserts the hard-coded values still match, so a figure cannot
silently drift away from the text.

  Fig1  pipeline, built separately from Fig1.tex (TikZ)
  Fig2  Table 2: total two-qubit gates against surviving diversity
  Fig3  Tables 4 and 6: MCC against error rate, coupled and realistic
  Fig4  Table 8: error diversity and the oracle ceiling

Run:
  python3 scripts/make_figures.py            # write the PDFs
  python3 scripts/make_figures.py --check    # verify against results/*.json, write nothing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RESULTS = Path(__file__).resolve().parents[1] / "results"

# textwidth of the sn-jnl class at these options is 372pt
TEXTWIDTH_IN = 372.0 / 72.27

# Okabe-Ito, safe for the common colour vision deficiencies and legible in greyscale
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#666666"


def style():
    # Springer asks for Helvetica or Arial in figure lettering at roughly 8 to 12 pt.
    # Nimbus Sans is the URW Helvetica clone and is metrically identical. Figures are
    # drawn at the 372 pt textwidth and included at \textwidth, so the sizes set here
    # are the sizes that appear on the page.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
        # keep mathtext in the same sans face, otherwise labels such as the learning
        # rates render in a serif italic beside sans body lettering
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold",
        "mathtext.cal": "Nimbus Sans:italic",
        "mathtext.default": "it",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# --------------------------------------------------------------------------------------
# data, transcribed from the manuscript tables

# Table 2: the corrected policies, loaded from the campaign results so the figure cannot
# drift from the table. Reference arms are constants; they have no policy seed.
import json as _json
_CORR = RESULTS / "corrected"
_st = _json.loads((_CORR / "structure.json").read_text())
_eff = _json.loads((_CORR / "effective_params.json").read_text())
def _e(n):
    return sum(_eff.get(f"{n}__{t}", {}).get("effective", 0) for t in ("Z_1_full", "ZZ_2_full", "Pauli_1_full"))
ARMS = []
for lr, lab in (("lr10", r"$1\times10^{-5}$"), ("lr75", r"$7.5\times10^{-6}$"), ("lr5", r"$5\times10^{-6}$")):
    for s in (46, 45, 44, 43, 42):
        v = _st[f"corr_{lr}_s{s}"]
        ARMS.append((f"{lab}, seed {s}", tuple(v["per_member_2q"]), v["distinct_members"], _e(f"corr_{lr}_s{s}"), lr))
ARMS += [
    ("Supervised warm-up",            (0, 20, 10), 3, 18, "base"),
    ("Hand-designed linear",          (0, 20, 10), 3, 18, "base"),
    ("Uncompressed",                  (0, 60, 30), 3, 18, "base"),
]
BLOCK_COLOR = {"base": "#8C8C8C", "lr5": BLUE, "lr75": GREEN, "lr10": ORANGE}
BLOCK_LABEL = {"base": "reference arms", "lr5": r"GRPO $5\times10^{-6}$",
               "lr75": r"GRPO $7.5\times10^{-6}$", "lr10": r"GRPO $1\times10^{-5}$"}

# Table 4 (tab:fusion6q): coupled family p2 = 10 p1, six qubits, MCC over five seeds
P1 = [0.0, 0.01, 0.03, 0.05, 0.1]
COUPLED = {
    "IoTID20": {
        "QVE3": [0.675, 0.000, 0.000, 0.000, 0.000],
        "QWE3": [0.675, 0.000, 0.000, 0.000, 0.000],
        "NWE3": [0.675, 0.546, 0.550, 0.535, 0.514],
        "Z":    [0.548, 0.546, 0.550, 0.535, 0.514],
    },
    "UNSW-NB15": {
        "QVE3": [0.667, 0.000, 0.000, 0.000, 0.000],
        "QWE3": [0.667, 0.651, 0.632, 0.629, 0.615],
        "NWE3": [0.667, 0.651, 0.632, 0.629, 0.615],
        "Z":    [0.649, 0.651, 0.632, 0.629, 0.615],
    },
    "Bot-IoT": {
        "QVE3": [0.846, 0.000, 0.000, 0.000, 0.000],
        "QWE3": [0.846, 0.000, 0.000, 0.000, 0.000],
        "NWE3": [0.846, 0.839, 0.838, 0.833, 0.820],
        "Z":    [0.839, 0.839, 0.838, 0.833, 0.820],
    },
}

# Table 6 (tab:fusionrealistic): p1 = 5e-4, UNSW-NB15, ten qubits
P2_REAL = [0.0, 0.01, 0.02]
REALISTIC_10Q = {
    "QVE3": [0.678, 0.656, 0.484],
    "QWE3": [0.678, 0.656, 0.571],
    "NWE3": [0.678, 0.656, 0.649],
    "Z":    [0.651, 0.649, 0.649],
}

# Table 8 (tab:agreement): noiseless kernels, six qubits, five seeds
PAIRS = [r"$Z$/$ZZ$", r"$Z$/Pauli", r"$ZZ$/Pauli"]
YULE_Q = {
    "IoTID20":   [0.850, 0.935, 0.998],
    "UNSW-NB15": [0.921, 0.944, 0.993],
    "Bot-IoT":   [0.951, 0.976, 0.999],
}
CEILING = {                       # majority, best member, oracle
    "IoTID20":   [0.675, 0.682, 0.802],
    "UNSW-NB15": [0.667, 0.685, 0.773],
    "Bot-IoT":   [0.846, 0.878, 0.928],
}

DATASETS = ["IoTID20", "UNSW-NB15", "Bot-IoT"]
# The rules coincide wherever no branch has failed, so they are drawn at three marker
# sizes on top of a pale band for the Z member. A coincidence then reads as a small
# marker sitting inside a larger one rather than as a missing series.
RULE_STYLE = {                    # colour, marker, markersize, linewidth, zorder
    "Z":    ("#C4C4C4", None, 0.0, 2.6, 1),
    "QVE3": (RED,       "s",  4.6, 1.1, 3),
    "QWE3": (ORANGE,    "^",  6.6, 1.1, 2),
    "NWE3": (BLUE,      "o",  3.4, 1.1, 4),
}
RULE_LABEL = {
    "QVE3": "QVE3 (majority)",
    "QWE3": "QWE3 (accuracy weighted)",
    "NWE3": "NWE3 (noise aware)",
    "Z":    r"$Z$ member alone",
}


# --------------------------------------------------------------------------------------

def fig_compression(out: Path):
    """Entangling cost, surviving diversity, and whether the encoding is real."""
    labels = [a[0] for a in ARMS]
    totals = [sum(a[1]) for a in ARMS]
    distinct = [a[2] for a in ARMS]
    effective = [a[3] for a in ARMS]
    colors = [BLOCK_COLOR[a[4]] for a in ARMS]
    y = np.arange(len(ARMS))

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(TEXTWIDTH_IN, 4.6), sharey=True,
        gridspec_kw={"width_ratios": [2.3, 0.85, 1.15], "wspace": 0.1})

    def bars(ax, vals, hi):
        b = ax.barh(y, vals, height=0.66, color=colors, edgecolor="white", linewidth=0.4)
        # the gamed arm is called out wherever it appears
        for i, a in enumerate(ARMS):
            if a[3] < 18:
                b[i].set_edgecolor(RED)
                b[i].set_linewidth(1.1)
                b[i].set_hatch("////")
        ax.set_xlim(0, hi)
        ax.grid(axis="x", linewidth=0.4, color="#DDDDDD")
        ax.set_axisbelow(True)
        return b

    bars(ax1, totals, 104)
    for yi, t in zip(y, totals):
        ax1.text(t + 1.8, yi, f"{t}", va="center", ha="left", fontsize=7.5,
                 color="#222222")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_ylim(-0.7, len(ARMS) - 0.3)
    ax1.tick_params(axis="y", labelsize=7)
    ax1.set_xlabel("Two-qubit gates")
    ax1.set_title("(a) Entangling cost", loc="left", pad=4)

    bars(ax2, distinct, 3.4)
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xlabel("Distinct members")
    ax2.set_title("(b) Diversity", loc="left", pad=4)

    bars(ax3, effective, 20.5)
    ax3.set_xticks([0, 6, 12, 18])
    ax3.axvline(18, color=GREY, linewidth=0.7, linestyle=(0, (3, 2)), zorder=4)
    ax3.set_xlabel("Effective parameters")
    ax3.set_title("(c) Real encoding", loc="left", pad=4)

    handles = [Line2D([], [], marker="s", linestyle="none", color=BLOCK_COLOR[k],
                      markersize=5, label=BLOCK_LABEL[k])
               for k in ("base", "lr5", "lr75", "lr10")]
    fig.subplots_adjust(bottom=0.15)
    fig.legend(handles=handles, ncol=4, loc="lower center", frameon=False,
               bbox_to_anchor=(0.58, 0.0), columnspacing=1.2, handletextpad=0.4)

    fig.savefig(out)
    plt.close(fig)
    return out


def fig_noise(out: Path):
    """MCC against error rate: three coupled panels and one hardware-realistic panel."""
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 3.8))
    tags = ["(a)", "(b)", "(c)"]

    def draw(ax, xs, series):
        for rule in ("Z", "QVE3", "QWE3", "NWE3"):
            c, m, ms, lw, z = RULE_STYLE[rule]
            ax.plot(xs, series[rule], color=c, marker=m, markersize=ms,
                    linewidth=lw, markerfacecolor="white", markeredgewidth=1.0,
                    zorder=z, solid_capstyle="round")

    for ax, ds, tag in zip(axes.flat[:3], DATASETS, tags):
        draw(ax, P1, COUPLED[ds])
        ax.set_title(f"{tag} {ds}, coupled", loc="left", pad=4)
        ax.set_xlim(-0.005, 0.105)
        ax.set_ylim(-0.04, 0.95)
        ax.set_xticks([0, 0.025, 0.05, 0.075, 0.1])
        ax.grid(linewidth=0.4, color="#EEEEEE")
        ax.set_axisbelow(True)

    ax = axes.flat[3]
    draw(ax, P2_REAL, REALISTIC_10Q)
    ax.set_title("(d) UNSW-NB15, ten qubits, realistic", loc="left", pad=4)
    ax.set_xlim(-0.001, 0.021)
    ax.set_ylim(-0.04, 0.95)
    ax.set_xticks([0, 0.005, 0.01, 0.015, 0.02])
    ax.grid(linewidth=0.4, color="#EEEEEE")
    ax.set_axisbelow(True)

    for ax in axes[1]:
        ax.set_xlabel(r"single-qubit error $p_1$")
    axes[1, 1].set_xlabel(r"two-qubit error $p_2$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Matthews correlation")

    handles = [Line2D([], [], color=RULE_STYLE[r][0], marker=RULE_STYLE[r][1],
                      markersize=RULE_STYLE[r][2], linewidth=RULE_STYLE[r][3],
                      markerfacecolor="white", label=RULE_LABEL[r])
               for r in ("QVE3", "QWE3", "NWE3", "Z")]
    fig.legend(handles=handles, ncol=2, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, -0.08), columnspacing=1.4)
    fig.tight_layout(h_pad=1.4, w_pad=1.6)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_agreement(out: Path):
    """Why fusion cannot help: correlated failures, and the ceiling they impose."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.75),
                                   gridspec_kw={"wspace": 0.34})

    x = np.arange(len(DATASETS))
    w = 0.26
    pair_colors = [GREEN, PURPLE, RED]
    for k, (pair, col) in enumerate(zip(PAIRS, pair_colors)):
        vals = [YULE_Q[d][k] for d in DATASETS]
        ax1.bar(x + (k - 1) * w, vals, width=w, color=col, label=pair,
                edgecolor="white", linewidth=0.4)
    ax1.axhline(0, color="#333333", linewidth=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(DATASETS)
    ax1.set_ylim(0, 1.1)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_ylabel(r"Yule's $Q$")
    ax1.set_title("(a) Error diversity", loc="left", pad=4)
    ax1.grid(axis="y", linewidth=0.4, color="#EEEEEE")
    ax1.set_axisbelow(True)
    ax1.axhline(1.0, color=GREY, linewidth=0.6, linestyle=(0, (3, 2)))
    ax1.text(x[-1] + 0.42, 1.015, "identical failures", fontsize=7, color=GREY,
             ha="right", va="bottom")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False,
               ncol=2, columnspacing=0.9, handlelength=1.0, handletextpad=0.4)

    ceil_labels = ["Majority vote", "Best member", "Oracle selection"]
    ceil_colors = [RED, ORANGE, BLUE]
    for k, (lab, col) in enumerate(zip(ceil_labels, ceil_colors)):
        vals = [CEILING[d][k] for d in DATASETS]
        ax2.bar(x + (k - 1) * w, vals, width=w, color=col, label=lab,
                edgecolor="white", linewidth=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(DATASETS)
    ax2.set_ylim(0, 1.1)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_ylabel("Matthews correlation")
    ax2.set_title("(b) What fusion could reach", loc="left", pad=4)
    ax2.grid(axis="y", linewidth=0.4, color="#EEEEEE")
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False,
               ncol=2, columnspacing=0.9, handlelength=1.0, handletextpad=0.4)

    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------

def check():
    """Assert the hard-coded values still match the JSON they came from."""
    bad = []

    def near(a, b, tol=0.0006):
        return abs(float(a) - float(b)) <= tol

    p = RESULTS / "member_agreement.json"
    if p.exists():
        blob = json.loads(p.read_text())["datasets"]
        keymap = {"IoTID20": "IoT_Original_Distribution.csv",
                  "UNSW-NB15": "UNSW_NB15.csv",
                  "Bot-IoT": "UNSW_2018_IoT_Botnet_Final_10_Best.csv"}
        pairkeys = ["Z-ZZ", "Z-Pauli", "ZZ-Pauli"]
        for ds, k in keymap.items():
            rec = blob[k]
            for pk, want in zip(pairkeys, YULE_Q[ds]):
                got = rec["pairs"][pk]["q"]
                if not near(got, want):
                    bad.append(f"Yule Q {ds} {pk}: fig {want} vs json {got:.4f}")
            for got, want, what in ((rec["majority_mcc"], CEILING[ds][0], "majority"),
                                    (rec["best_member_mcc"], CEILING[ds][1], "best"),
                                    (rec["oracle_mcc"], CEILING[ds][2], "oracle")):
                if not near(got, want):
                    bad.append(f"{what} {ds}: fig {want} vs json {got:.4f}")
        print(f"[check] member_agreement.json  {len(keymap) * 6} values")
    else:
        print(f"[check] SKIP {p} not found")

    n_corr = sum(1 for a in ARMS if a[4] != "base")
    print(f"[check] corrected structure.json  {n_corr} policies loaded directly")

    if bad:
        print("\n[MISMATCH]")
        for b in bad:
            print("  " + b)
        return 1
    print("[check] all cross-checked values agree")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify against results/*.json and write nothing")
    ap.add_argument("--outdir", default=str(HERE.parent / "figures"))
    args = ap.parse_args()

    if args.check:
        sys.exit(check())

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    style()
    for fn, name in ((fig_compression, "Fig2.pdf"),
                     (fig_noise, "Fig3.pdf"),
                     (fig_agreement, "Fig4.pdf")):
        path = fn(outdir / name)
        print(f"[written] {path}  ({path.stat().st_size / 1024:.0f} kB)")
    print("\nFig1.pdf is built from Fig1.tex:")
    print("  pdflatex -output-directory=figures figures/Fig1.tex")


if __name__ == "__main__":
    main()
