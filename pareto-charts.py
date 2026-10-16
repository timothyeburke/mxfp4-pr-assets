#!/usr/bin/env python3
"""Pareto accuracy-vs-size figures from the single-campaign sweep (pareto-sweep-results.tsv).

Issue figure: 27B (dense, the clean case) - the empty cell of the 2x3 grid.
PR figure: all three models, one panel each.

Design: file size (GB, log) on x, wikitext-2 PPL on y (lower is better).
Standard quants are gray; the two mxfp4 arms are green (W4A4 hollow, W4A8 star)
with an arrow from W4A4 to W4A8 showing the move.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv, os, math

OUT = os.path.dirname(os.path.abspath(__file__))
MX = "#76B900"     # mxfp4
OTHER = "#9AA4B2"  # standard quants
TEXT = "#1F2933"
MUTED = "#6B7280"
ARROW = "#C0392B"  # the improvement move

plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": TEXT,
    "axes.edgecolor": "#D5DAE1", "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#ECEFF3", "grid.linewidth": 1.0,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

def load(path):
    d = {}
    with open(path) as fh:
        next(fh, None)  # skip header
        for line in fh:
            p = line.split()
            if len(p) < 4:
                continue
            # model first; label is p[1]; size last; ppl second-to-last
            d.setdefault(p[0], {})[p[1]] = (float(p[-1]), float(p[-2]))
    return d

MX_ARMS = ("mxfp4-w4a4", "mxfp4-w4a8")
STD = ("Q4_0", "Q4_1", "Q4_K_M", "Q5_1", "Q8_0")

def pareto_panel(ax, rows, title):
    """rows: {quant: (size_gb, ppl)}"""
    ax.set_yscale("linear")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("file size (GB)", fontsize=8)
    ax.set_ylabel("PPL (lower better)", fontsize=8)
    # linear x: all size ranges are well under a decade
    # standard quants
    for q in STD:
        if q in rows:
            s, p = rows[q]
            ax.scatter([s], [p], s=46, c=OTHER, marker="o", zorder=3, edgecolors="white", linewidths=0.6)
            ax.annotate(q, (s, p), textcoords="offset points", xytext=(6, 0),
                        fontsize=7, color=MUTED, ha="left", va="center")
    # mxfp4 arms
    s4, p4 = rows["mxfp4-w4a4"]
    s8, p8 = rows["mxfp4-w4a8"]
    ax.scatter([s4], [p4], s=120, c="white", marker="o", edgecolors=MX, linewidths=2.2, zorder=5)
    ax.scatter([s8], [p8], s=230, c=MX, marker="*", zorder=6, edgecolors="white", linewidths=0.8)
    # arrow W4A4 -> W4A8 (down = lower PPL = better)
    ax.annotate("", xy=(s8, p8), xytext=(s4, p4),
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.8,
                               connectionstyle="arc3,rad=0.15"), zorder=4)
    ax.annotate("W4A4 (today)", (s4, p4), textcoords="offset points", xytext=(-8, 8),
                fontsize=7.5, color=ARROW, ha="right")
    ax.annotate("W4A8 (this PR)", (s8, p8), textcoords="offset points", xytext=(10, -12),
                fontsize=7.5, color=MX, ha="left", fontweight="bold")
    # x ticks: log scale, range < 1 decade -> plain-number labels
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    sizes = [rows[q][0] for q in rows]
    ax.set_xlim(min(sizes) * 0.97, max(sizes) * 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}" if v < 10 else f"{v:.1f}"))
    # y range
    vals = [rows[q][1] for q in rows]
    ymin, ymax = min(vals), max(vals)
    pad = (ymax - ymin) * 0.18
    ax.set_ylim(ymin - pad, ymax + pad)
    return ax

def issue_27b(d, out):
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.4))
    pareto_panel(ax, d["27B"], "Qwen3.8-27B, 2x RTX 5070 Ti (this PR)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)

def all_models(d, out):
    fig, axes = plt.subplots(1, 3, figsize=(5.4 * 3, 4.3))
    names = {"0.8B": "Qwen3.5-0.8B", "27B": "Qwen3.8-27B", "35B": "Qwen3.6-35B-A3B"}
    for ax, m in zip(axes, ("0.8B", "27B", "35B")):
        pareto_panel(ax, d[m], names[m] + ("  (MoE)" if m == "35B" else ""))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)

if __name__ == "__main__":
    d = load("pareto-sweep-results.tsv")
    issue_27b(d, os.path.join(OUT, "pareto-27b.png"))
    all_models(d, os.path.join(OUT, "pareto-all.png"))
