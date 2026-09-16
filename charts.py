#!/usr/bin/env python3
"""
mxfp4 PR charts - single generator for all PR figures.

One design language, shared by every chart:
  - shared PALETTE (mxfp4 = accent, family = blue, refs = gray)
  - shared figure/axis styling (new_figure / style_axes / save)
  - minimal in-image text: small labels only; prose lives in the PR body
Data is the fresh full-imatrix, all-chunks run (see mxfp4-imatrix-accuracy.md).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ design
PALETTE = {
    "accent":  "#E4572E",   # mxfp4 (the subject)
    "family":  "#5B8DEF",   # 4/5-bit family
    "ref":     "#A6ADBB",   # bf16 / Q8_0 references
    "text":    "#1F2933",
    "muted":   "#6B7280",
    "grid":    "#ECEFF3",
    "spine":   "#D5DAE1",
    "bg":      "#FFFFFF",
}
FONT = {"family": "DejaVu Sans", "size": 12}

plt.rcParams.update({
    "font.family": FONT["family"],
    "text.color": PALETTE["text"],
    "axes.edgecolor": PALETTE["spine"],
    "axes.labelcolor": PALETTE["text"],
    "xtick.color": PALETTE["muted"],
    "ytick.color": PALETTE["muted"],
    "axes.grid": True,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 1.0,
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor": PALETTE["bg"],
    "savefig.facecolor": PALETTE["bg"],
    "svg.fonttype": "none",
})


def new_figure(nrows=1, ncols=1, w=6.0, h=4.4, sharey=False):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    return fig, axes.flatten()


def style_axes(ax, title=None, xlabel=None, ylabel=None, xlog=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(PALETTE["spine"])
    ax.spines["bottom"].set_color(PALETTE["spine"])
    ax.tick_params(length=0)
    ax.grid(axis="both", linestyle="-", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=13, color=PALETTE["text"], pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=PALETTE["muted"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=PALETTE["muted"])
    if xlog:
        ax.set_xscale("log")
    return ax


def legend_items(ax, items, **kw):
    ax.legend(handles=items, **kw)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------------------ data
# fresh full-imatrix, all-chunks. (type, size_GB, imx_ppl, no_imx_ppl or None, is_ref)
PPL = {
    "0.8B": [
        ("bf16", 1.6, 15.129, None, True),
        ("Q8_0", 0.8, 15.162, None, True),
        ("q5_1", 0.6, 15.350, 15.204, False),
        ("q4_1", 0.5, 15.786, 15.811, False),
        ("iq4xs", 0.5, 15.870, 15.962, False),
        ("q4ks", 0.5, 15.999, 15.811, False),
        ("mxfp4", 0.6, 16.181, 17.489, False),
        ("q4_0", 0.5, 18.444, 19.107, False),
        ("q3ks", 0.4, 19.601, 21.516, False),
    ],
    "27B": [
        ("q4ks", 15.6, 6.294, 6.326, False),
        ("q4_1", 17.1, 6.342, 6.320, False),
        ("mxfp4", 15.7, 6.364, 6.441, False),
        ("q5ks", 18.7, 6.375, 6.495, False),
        ("q5_1", 20.3, 6.417, 6.505, False),
        ("Q8_0", 28.6, 6.434, None, True),
        ("bf16", 53.8, 6.435, None, True),
        ("iq4xs", 15.1, 6.438, 6.413, False),
        ("q4_0", 15.5, 6.572, 6.475, False),
        ("q3ks", 12.1, 6.779, 7.375, False),
    ],
    "35B": [
        ("Q8_0", 36.9, 5.740, None, True),
        ("bf16", 69.4, 5.745, None, True),
        ("q5_1", 26.1, 5.756, 5.790, False),
        ("q5ks", 24.0, 5.785, 5.834, False),
        ("q4_1", 21.8, 5.804, 5.887, False),
        ("q4ks", 19.9, 5.808, 5.825, False),
        ("q4_0", 19.8, 5.839, 5.886, False),
        ("iq4xs", 18.7, 5.859, 5.849, False),
        ("mxfp4", 19.0, 5.934, 5.999, False),
        ("mxfp4-moe", 19.8, 5.776, None, False),
        ("q3ks", 15.2, 6.194, 6.525, False),
    ],
}


def scatter_quant_points(ax, rows):
    """Plot one model's quants: family (blue), mxfp4 (accent, larger), refs (gray, diamond); label each point."""
    # small per-point label offsets (dx, dy in points) to keep the tight 4-bit cluster legible
    off = {
        "bf16": (7, -15), "Q8_0": (7, -15), "q3ks": (7, 7),
        "q5_1": (7, 9), "q5ks": (7, 9), "q4_1": (11, -13),
        "q4ks": (6, 8), "q4_0": (7, -14), "iq4xs": (12, -1),
        "mxfp4": (14, 4), "moe-recipe": (9, -16),
    }
    fam_x, fam_y, acc_x, acc_y, moe_x, moe_y, ref_x, ref_y = [], [], [], [], [], [], [], []
    pts = []
    for (name, size, imx, noimx, is_ref) in rows:
        disp = "moe-recipe" if name == "mxfp4-moe" else name
        pts.append((size, imx, disp, is_ref, name in ("mxfp4", "mxfp4-moe")))
        if is_ref:
            ref_x.append(size); ref_y.append(imx)
        elif name == "mxfp4":
            acc_x.append(size); acc_y.append(imx)
        elif name == "mxfp4-moe":
            moe_x.append(size); moe_y.append(imx)
        else:
            fam_x.append(size); fam_y.append(imx)
    ax.scatter(fam_x, fam_y, s=42, c=PALETTE["family"], zorder=3, edgecolors="white", linewidths=0.6)
    ax.scatter(ref_x, ref_y, s=54, marker="D", c=PALETTE["ref"], zorder=3, edgecolors="white", linewidths=0.6)
    ax.scatter(acc_x, acc_y, s=120, c=PALETTE["accent"], zorder=5, edgecolors="white", linewidths=1.0)
    ax.scatter(moe_x, moe_y, s=120, marker="D", c=PALETTE["accent"], zorder=5, edgecolors="white", linewidths=1.0)
    for (x, y, disp, is_ref, is_mx) in pts:
        dx, dy = off.get(disp, (7, 8))
        col = PALETTE["muted"] if is_ref else (PALETTE["accent"] if is_mx else PALETTE["text"])
        ax.annotate(disp, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=7, color=col, ha="left", va="bottom", zorder=6)
    return [
        Line2D([0], [0], marker="o", ls="", ms=9, mec="white", mfc=PALETTE["accent"], label="mxfp4 (all)"),
        Line2D([0], [0], marker="D", ls="", ms=9, mec="white", mfc=PALETTE["accent"], label="mxfp4 (MoE recipe)"),
        Line2D([0], [0], marker="o", ls="", ms=7, mec="white", mfc=PALETTE["family"], label="4/5-bit family"),
        Line2D([0], [0], marker="D", ls="", ms=8, mec="white", mfc=PALETTE["ref"], label="ref (bf16 / Q8_0)"),
    ]


def ppl_vs_size():
    fig, axgrid = new_figure(1, 3, w=5.4, h=4.3)
    handles = None
    for ax, (model, rows) in zip(axgrid, PPL.items()):
        style_axes(ax, title=model, xlabel="file size (GB)", ylabel="PPL")
        h = scatter_quant_points(ax, rows)
        if handles is None:
            handles = h
        ax.set_xlim(min(r[1] for r in rows) * 0.9, max(r[1] for r in rows) * 1.08)
        ymin, ymax = min(r[2] for r in rows), max(r[2] for r in rows)
        pad = (ymax - ymin) * 0.15
        ax.set_ylim(ymin - pad, ymax + pad)
    legend_items(axgrid[0], handles, loc="lower right", frameon=False, fontsize=9,
                             ncol=1, handlelength=1.4, borderaxespad=0.6)
    save(fig, "ppl-vs-size.png")
    return


# ------------------------------------------------------------------ KV cache
KV_TYPES = ["f16", "q4_0", "q4_1", "q5_1", "mxfp4"]
KV = {
    "0.8B": {"mem": {"f16": 1.14, "q4_0": 0.32, "q4_1": 0.36, "q5_1": 0.43, "mxfp4": 0.30},
             "tg":  {"f16": 434.4, "q4_0": 411.2, "q4_1": 413.6, "q5_1": 414.7, "mxfp4": 412.9}},
    "27B":  {"mem": {"f16": 6.10, "q4_0": 1.72, "q4_1": 1.91, "q5_1": 2.29, "mxfp4": 1.62},
             "tg":  {"f16": 42.6, "q4_0": 42.1, "q4_1": 42.1, "q5_1": 42.1, "mxfp4": 42.2}},
    "35B":  {"mem": {"f16": 1.91, "q4_0": 0.54, "q4_1": 0.60, "q5_1": 0.72, "mxfp4": 0.51},
             "tg":  {"f16": 183.9, "q4_0": 178.7, "q4_1": 179.3, "q5_1": 179.1, "mxfp4": 179.1}},
}

def kv_color(t):
    if t == "f16":
        return PALETTE["ref"]
    if t == "mxfp4":
        return PALETTE["accent"]
    return PALETTE["family"]

def kv_cache():
    fig, ax = new_figure(2, 3, w=5.0, h=3.5)
    for col, (model, d) in enumerate(KV.items()):
        axm = ax[col]
        style_axes(axm, title=model, ylabel="KV mem (GiB @100k)" if col == 0 else None)
        bm = axm.bar(KV_TYPES, [d["mem"][t] for t in KV_TYPES], color=[kv_color(t) for t in KV_TYPES], width=0.72)
        axm.bar_label(bm, fmt="%.2f", fontsize=7, color=PALETTE["text"], padding=2)
        axm.set_ylim(0, max(d["mem"].values()) * 1.22)
        axt = ax[3 + col]
        style_axes(axt, ylabel="decode (t/s)" if col == 0 else None)
        bt = axt.bar(KV_TYPES, [d["tg"][t] for t in KV_TYPES], color=[kv_color(t) for t in KV_TYPES], width=0.72)
        axt.bar_label(bt, fmt="%.1f", fontsize=7, color=PALETTE["text"], padding=2)
        axt.set_ylim(0, max(d["tg"].values()) * 1.22)
    save(fig, "kv-cache.png")

# ------------------------------------------------------------------ KL + top-p
# per quant: (imx_kld, imx_topp, plain_kld, plain_topp); q8 has no plain
KL_ORDER = ["q8", "q5_1", "q5ks", "q4_1", "q4ks", "iq4xs", "mxfp4", "q4_0", "q3ks"]
KL = {
    "0.8B": {
        "q8":    (0.001274, 98.029, None, None),
        "q5_1":  (0.015941, 93.279, 0.025742, 91.611),
        "q5ks":  (0.018677, 92.763, 0.026215, 91.246),
        "q4_1":  (0.052838, 88.337, 0.093422, 84.593),
        "q4ks":  (0.052137, 88.432, 0.076593, 86.340),
        "iq4xs": (0.056000, 88.111, 0.069437, 86.883),
        "mxfp4": (0.149358, 81.307, 0.187225, 78.825),
        "q4_0":  (0.113817, 83.581, 0.143692, 81.412),
        "q3ks":  (0.278512, 74.846, 0.376418, 71.496),
    },
    "27B": {
        "q8":    (0.009147, 98.674, None, None),
        "q5_1":  (0.025711, 96.304, 0.034100, 95.434),
        "q5ks":  (0.026647, 96.077, 0.032669, 95.538),
        "q4_1":  (0.044071, 94.225, 0.068571, 92.060),
        "q4ks":  (0.046680, 93.995, 0.059856, 92.591),
        "iq4xs": (0.039574, 94.317, 0.049913, 93.341),
        "mxfp4": (0.089838, 90.217, 0.101166, 89.239),
        "q4_0":  (0.058929, 92.702, 0.069682, 91.645),
        "q3ks":  (0.114200, 87.891, 0.173826, 85.424),
    },
    "35B": {
        "q8":    (0.005252, 97.400, None, None),
        "q5_1":  (0.013733, 95.239, 0.020446, 94.306),
        "q5ks":  (0.014901, 95.099, 0.021629, 94.104),
        "q4_1":  (0.029135, 93.073, 0.052624, 90.574),
        "q4ks":  (0.029115, 93.066, 0.043272, 91.421),
        "iq4xs": (0.029952, 92.986, 0.038388, 91.981),
        "mxfp4": (0.068176, 89.343, 0.087843, 87.920),
        "q4_0":  (0.046439, 91.255, 0.055915, 90.307),
        "q3ks":  (0.106026, 86.843, 0.151525, 84.141),
    },
}

def _kl_color(q):
    if q == "q8":
        return PALETTE["ref"]
    if q == "mxfp4":
        return PALETTE["accent"]
    return PALETTE["family"]

def _plot_kl_panel(ax, d, key):
    xs = list(range(len(KL_ORDER)))
    imx_x, imx_y = [], []
    pl_x, pl_y = [], []
    cols = []
    for i, q in enumerate(KL_ORDER):
        ik, it, pk, pt = d[q]
        yv = ik if key == "kld" else it
        imx_x.append(i - 0.18); imx_y.append(yv); cols.append(_kl_color(q))
        if pk is not None:
            pl_x.append(i + 0.18); pl_y.append(pk if key == "kld" else pt)
    ax.scatter(imx_x, imx_y, s=64, c=cols, zorder=5, edgecolors="white", linewidths=0.8)
    if pl_x:
        ax.scatter(pl_x, pl_y, s=64, facecolors="none", edgecolors=cols[:len(pl_x)], linewidths=1.4, zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels(KL_ORDER, fontsize=8)
    ax.set_xlim(-0.6, len(KL_ORDER) - 0.4)
    ys = imx_y + pl_y
    ymin, ymax = min(ys), max(ys)
    pad = (ymax - ymin) * 0.18
    ax.set_ylim(ymin - pad, ymax + pad)

def kl_top_p():
    fig, ax = new_figure(2, 3, w=5.0, h=3.6)
    for col, (model, d) in enumerate(KL.items()):
        style_axes(ax[col], title=model, ylabel="Mean KLD (lower is better)")
        _plot_kl_panel(ax[col], d, "kld")
        style_axes(ax[3 + col], ylabel="Same top-p % (higher is better)")
        _plot_kl_panel(ax[3 + col], d, "topp")
    handles = [Line2D([0], [0], marker="o", ls="", ms=8, mec="white", mfc=PALETTE["accent"], label="imatrix"),
               Line2D([0], [0], marker="o", ls="", ms=8, mfc="none", mec=PALETTE["family"], label="no imatrix"),
               Line2D([0], [0], marker="o", ls="", ms=8, mec="white", mfc=PALETTE["ref"], label="ref (Q8_0)")]
    legend_items(ax[0], handles, loc="upper right", frameon=False, fontsize=8, ncol=3,
                             handlelength=1.2, columnspacing=0.8, borderaxespad=0.4)
    save(fig, "kl-top-p.png")

if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "ppl"):
        ppl_vs_size()
    if which in ("all", "kv"):
        kv_cache()
    if which in ("all", "kl"):
        kl_top_p()
