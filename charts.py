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
import math
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ design
PALETTE = {
    "mx":      "#76B900",   # mxfp4 (NVIDIA green)
    "other":   "#9AA4B2",   # everything else (all refs: bf16, Q8_0, q4*, q5*, q3*)
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
        ax.tick_params(which="minor", labelbottom=False, labelleft=False)
    return ax


def legend_items(ax, items, **kw):
    ax.legend(handles=items, **kw)


def print_tick_labels(fig):
    """Debug: print the tick labels exactly as rendered, per axis."""
    fig.canvas.draw()
    for i, ax in enumerate(fig.get_axes()):
        xl = [t.get_text() for t in ax.get_xticklabels()]
        yl = [t.get_text() for t in ax.get_yticklabels()]
        print(f"  [axis {i}] x={xl} y={yl}")


def save(fig, name):
    print_tick_labels(fig)
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
        ("mxfp4-moe", 19.8, 5.776, 5.798, False),
        ("q3ks", 15.2, 6.194, 6.525, False),
    ],
}


def scatter_quant_points(ax, rows):
    """Plot one model's quants: mxfp4 (NVIDIA green), everything else gray; imx filled, no-imx hollow."""
    for (name, size, imx, noimx, is_ref) in rows:
        disp = "mxfp4_moe" if name == "mxfp4-moe" else name
        is_mx = name in ("mxfp4", "mxfp4-moe")
        col = PALETTE["mx"] if is_mx else PALETTE["other"]
        marker = "D" if name == "mxfp4-moe" else "o"
        s = 110 if is_mx else 46
        ax.scatter([size], [imx], s=s, c=col, marker=marker, zorder=5, edgecolors="white", linewidths=0.8)
        if noimx is not None:
            ax.scatter([size], [noimx], s=s, facecolors="none", edgecolors=col, linewidths=1.6, marker=marker, zorder=4)
            ax.annotate(disp, (size, noimx), textcoords="offset points", xytext=(-7, 0),
                        fontsize=7, color=(PALETTE["mx"] if is_mx else PALETTE["muted"]),
                        ha="right", va="center", zorder=6)
        ax.annotate(disp, (size, imx), textcoords="offset points", xytext=(7, 0),
                    fontsize=7, color=(PALETTE["mx"] if is_mx else PALETTE["muted"]),
                    ha="left", va="center", zorder=6)
    items = [Line2D([0], [0], marker="o", ls="", ms=9, mec="white", mfc=PALETTE["mx"], label="mxfp4")]
    if any(r[0] == "mxfp4-moe" for r in rows):
        items.append(Line2D([0], [0], marker="D", ls="", ms=9, mec="white", mfc=PALETTE["mx"], label="mxfp4_moe"))
    items.append(Line2D([0], [0], marker="o", ls="", ms=7, mec="white", mfc=PALETTE["other"], label="others"))
    items.append(Line2D([0], [0], marker="o", ls="", ms=7, mec=PALETTE["other"], mfc="none", label="no imatrix"))
    items.append(Line2D([0], [0], color="#000000", ls="--", lw=1.2, alpha=0.1, label="bf16 (ref)"))
    ax.legend(handles=items, loc="upper right", frameon=False, fontsize=8, ncol=1, handlelength=1.4, borderaxespad=0.5)
    return


def ppl_vs_size():
    fig, axgrid = new_figure(1, 3, w=5.4, h=4.3)
    for ax, (model, rows) in zip(axgrid, PPL.items()):
        style_axes(ax, title=model, xlabel="file size (GB)", ylabel="PPL", xlog=True)
        plot_rows = [r for r in rows if r[0] not in ("bf16", "q3ks", "Q8_0")]
        scatter_quant_points(ax, plot_rows)
        bf16 = next((r[2] for r in rows if r[0] == "bf16"), None)
        if bf16 is not None:
            ax.axhline(bf16, color="#000000", linestyle="--", linewidth=1.2, alpha=0.1, zorder=2)
        ax.set_xlim(min(r[1] for r in plot_rows) * 0.98, max(r[1] for r in plot_rows) * 1.06)
        from matplotlib.ticker import FuncFormatter, MaxNLocator
        # log scale, but the range spans < 1 decade: place regular-number ticks with a linear-style locator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        xspan = max(r[1] for r in plot_rows) - min(r[1] for r in plot_rows)
        nd = 2 if xspan < 1 else 0
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _, nd=nd: f"{v:.{nd}f}" if v < 10 else f"{v:.0f}"))
        allvals = [r[2] for r in plot_rows] + [r[3] for r in plot_rows if r[3] is not None]
        ymin, ymax = min(allvals), max(allvals)
        pad = (ymax - ymin) * 0.15
        # round the top up to the next 0.1 so no dot sits on (or above) the axis edge
        top = math.ceil(ymax * 10 - 1e-9) / 10
        if top <= ymax + 1e-9:
            top += 0.1
        ax.set_ylim(ymin - pad, top)
    save(fig, "ppl-vs-size.png")
    return


# ------------------------------------------------------------------ KV cache
KV_TYPES = ["f16", "q4_0", "q4_1", "q5_1", "mxfp4"]
KV = {
    "0.8B": {"mem": {"f16": 1.14, "q4_0": 0.32, "q4_1": 0.36, "q5_1": 0.43, "mxfp4": 0.30},
             "tg":  {"f16": 398.0, "q4_0": 397.2, "q4_1": 393.5, "q5_1": 401.2, "mxfp4": 374.3},
             "cpu9900": {"f16": 47.4, "q4_0": 40.0, "q4_1": 41.1, "q5_1": 47.9, "mxfp4": 47.8},
             "ppl": {"f16": 16.1832, "q4_0": 16.3717, "q4_1": 16.3289, "q5_1": 16.2197, "mxfp4": 16.3692}},
    "27B":  {"mem": {"f16": 6.10, "q4_0": 1.72, "q4_1": 1.91, "q5_1": 2.29, "mxfp4": 1.62},
             "tg":  {"f16": 41.4, "q4_0": 41.3, "q4_1": 41.6, "q5_1": 41.7, "mxfp4": 41.3},
             "cpu9900": {"f16": 2.0, "q4_0": 1.8, "q4_1": 1.1, "q5_1": 1.9, "mxfp4": 2.2},
             "ppl": {"f16": 6.3536, "q4_0": 6.3744, "q4_1": 6.3671, "q5_1": 6.3568, "mxfp4": 6.4157}},
    "35B":  {"mem": {"f16": 1.91, "q4_0": 0.54, "q4_1": 0.60, "q5_1": 0.72, "mxfp4": 0.51},
             "tg":  {"f16": 181.2, "q4_0": 175.2, "q4_1": 172.2, "q5_1": 174.7, "mxfp4": 175.6},
             "cpu9900": {"f16": 12.5, "q4_0": 11.7, "q4_1": 12.9, "q5_1": 12.1, "mxfp4": 12.6},
             "ppl": {"f16": 5.9285, "q4_0": 5.9429, "q4_1": 5.9453, "q5_1": 5.9310, "mxfp4": 5.9504}},
}

def kv_color(t):
    return PALETTE["mx"] if t == "mxfp4" else PALETTE["other"]

def _speed_row(ax, vals, ylabel, log=True):
    style_axes(ax, ylabel=ylabel)
    b = ax.bar(KV_TYPES, [vals[t] for t in KV_TYPES], color=[kv_color(t) for t in KV_TYPES], width=0.72)
    ax.bar_label(b, fmt="%.1f", fontsize=7, color=PALETTE["text"], padding=2)
    if log:
        ax.set_yscale("log")
        ax.tick_params(which="minor", labelleft=False, labelbottom=False)
        ax.set_ylim(min(vals.values()) * 0.9, max(vals.values()) * 1.08)
        from matplotlib.ticker import FuncFormatter, MaxNLocator
        # log scale, but the range spans < 1 decade: place regular-number ticks with a linear-style locator
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    else:
        vmin, vmax = min(vals.values()), max(vals.values())
        ax.set_ylim(vmin - (vmax - vmin) * 0.9, vmax + (vmax - vmin) * 0.15)

def kv_cache():
    fig, ax = new_figure(5, 3, w=5.0, h=3.0)
    for col, (model, d) in enumerate(KV.items()):
        axm = ax[col]
        style_axes(axm, title=model, ylabel="KV mem (GiB @100k)" if col == 0 else None)
        bm = axm.bar(KV_TYPES, [d["mem"][t] for t in KV_TYPES], color=[kv_color(t) for t in KV_TYPES], width=0.72)
        axm.bar_label(bm, fmt="%.2f", fontsize=7, color=PALETTE["text"], padding=2)
        axm.set_ylim(0, max(d["mem"].values()) * 1.22)
        _speed_row(ax[3 + col], d["tg"], "GPU decode (t/s)" if col == 0 else None)
        _speed_row(ax[6 + col], d["cpu9900"], "9900X tg32 (t/s)" if col == 0 else None, log=False)
        axp = ax[9 + col]
        style_axes(axp, ylabel="PPL (mxfp4 weights, 72 chunks)" if col == 0 else None)
        f16p = d["ppl"]["f16"]
        kv4 = [t for t in KV_TYPES if t != "f16"]
        bp = axp.bar(kv4, [d["ppl"][t] for t in kv4], color=[kv_color(t) for t in kv4], width=0.6)
        axp.bar_label(bp, fmt="%.4f", fontsize=7, color=PALETTE["text"], padding=2)
        axp.plot([-0.55, 3.55], [f16p, f16p], color="#000000", ls="--", lw=1.2, alpha=0.3, zorder=2)
        axp.text(-0.55, f16p, f"  f16 {f16p:.4f}", fontsize=6.5, color=PALETTE["muted"], va="bottom")
        vmin, vmax = min(d["ppl"].values()), max(d["ppl"].values())
        axp.set_ylim(vmin * 0.999, vmax + (vmax - vmin) * 0.45)
        axu = ax[12 + col]
        style_axes(axu, ylabel="KLD vs BF16 base (mxfp4 KV)" if col == 0 else None)
        c0, e, u = KVDATA[model]["kld"]
        bu = axu.bar(["OCP e_base", "UOS 7.25"], [e, u], color=[PALETTE["other"], PALETTE["mx"]], width=0.5)
        axu.bar_label(bu, fmt="%.6f", fontsize=7, color=PALETTE["text"], padding=2)
        axu.plot([-0.5, 0.5], [c0, c0], color="#000000", ls="--", lw=1.2, alpha=0.3, zorder=2)
        axu.set_ylim(0, max(c0, e, u) * 1.15)
    save(fig, "kv-cache.png")

# ------------------------------------------------------------------ KL + top-p
# per quant: (imx_kld, imx_topp, plain_kld, plain_topp); q8 has no plain
KL_ORDER = ["q8", "q5_1", "q5ks", "q4_1", "q4ks", "iq4xs", "mxfp4", "mxfp4_moe", "q4_0", "q3ks"]
KL = {
    "0.8B": {
        "q8":    (0.001274, 98.029, None, None),
        "q5_1":  (0.015941, 93.279, 0.025742, 91.611),
        "q5ks":  (0.018677, 92.763, 0.026215, 91.246),
        "q4_1":  (0.052838, 88.337, 0.093422, 84.593),
        "q4ks":  (0.052137, 88.432, 0.076593, 86.340),
        "iq4xs": (0.056000, 88.111, 0.069437, 86.883),
        "mxfp4": (0.149358, 81.307, 0.187225, 78.825),
        "mxfp4_moe": (None, None, None, None),
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
        "mxfp4_moe": (None, None, None, None),
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
        "mxfp4_moe": (0.028607, 93.355, 0.029414, 93.289),
        "q4_0":  (0.046439, 91.255, 0.055915, 90.307),
        "q3ks":  (0.106026, 86.843, 0.151525, 84.141),
    },
}

def _kl_color(q):
    return PALETTE["mx"] if q in ("mxfp4", "mxfp4_moe") else PALETTE["other"]


def _plot_kl_panel(ax, d, key, order, ymin0=False):
    xs = np.arange(len(order))
    w = 0.4
    imx_x, imx_vals, cols = [], [], []
    noimx_x, noimx_vals, nocols = [], [], []
    for i, q in enumerate(order):
        ik, it, pk, pt = d[q]
        if ik is not None or it is not None:
            imx_x.append(i - w/2); imx_vals.append(ik if key == "kld" else it); cols.append(_kl_color(q))
        if pk is not None or pt is not None:
            noimx_x.append(i + w/2); noimx_vals.append(pk if key == "kld" else pt); nocols.append(_kl_color(q))
    ax.bar(imx_x, imx_vals, width=w, color=cols, zorder=3)
    if noimx_vals:
        ax.bar(noimx_x, noimx_vals, width=w, color="none", hatch="//", edgecolor=nocols, linewidth=1.0, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(order, fontsize=8, rotation=-90, ha="left")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ys = imx_vals + [v for v in noimx_vals if v is not None]
    ymin, ymax = (0.0 if ymin0 else min(ys)), max(ys)
    pad = (ymax - ymin) * 0.18
    ax.set_ylim(ymin if ymin0 else (ymin - pad), ymax + pad)

def kl_top_p():
    fig, ax = new_figure(2, 3, w=5.0, h=3.6)
    for col, (model, d) in enumerate(KL.items()):
        order = [q for q in KL_ORDER if any(v is not None for v in d[q])]
        style_axes(ax[col], title=model, ylabel="Mean KLD (lower is better)")
        _plot_kl_panel(ax[col], d, "kld", order, ymin0=True)
        style_axes(ax[3 + col], ylabel="Same top-p % (higher is better)")
        _plot_kl_panel(ax[3 + col], d, "topp", order)
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=PALETTE["mx"], edgecolor="white", label="mxfp4 (imatrix)"),
        Patch(facecolor="none", edgecolor=PALETTE["mx"], hatch="//", label="mxfp4 (no imatrix)"),
        Patch(facecolor=PALETTE["other"], edgecolor="white", label="3/4/5-bit (imatrix)"),
        Patch(facecolor="none", edgecolor=PALETTE["other"], hatch="//", label="3/4/5-bit (no imatrix)"),
    ]
    legend_items(ax[0], handles, loc="upper right", frameon=False, fontsize=8, ncol=2, handlelength=1.2, columnspacing=0.8, borderaxespad=0.4)
    save(fig, "kl-top-p.png")

# ------------------------------------------------------------------ W4A8 vs W4A4
# (ppl_w4a4, ppl_w4a8, pp_w4a4, pp_w4a8, tg_w4a4, tg_w4a8, kld_w4a4, kld_w4a8, topp_w4a4, topp_w4a8) for the imx file
# PPL/pp/tg from the 2x 5060 Ti collection; KLD/top-p from the 72-chunk KLD round (same files, f16 KV)
W4A5 = {
    "0.8B": {"w4a4_old": (21.390, 0.407816, 69.972), "w4a4_uos": (20.044, 0.354800, 71.753),
             "w4a8_256": (16.179, 0.149368, 81.280), "w4a8_343": (16.180, 0.149440, 81.243),
             "w4a8_464": (16.183, 0.149191, 81.283)},
    "27B":  {"w4a4_old": (6.559, 0.183153, 84.010), "w4a4_uos": (6.498, 0.182952, 84.523),
             "w4a8_256": (6.357, 0.089655, 90.138), "w4a8_343": (6.363, 0.089431, 90.176),
             "w4a8_464": (6.354, 0.089920, 90.182)},
    "35B":  {"w4a4_old": (6.467, 0.169050, 82.510), "w4a4_uos": (6.346, 0.149619, 83.785),
             "w4a8_256": (5.933, 0.068153, 89.276), "w4a8_343": (5.930, 0.068055, 89.301),
             "w4a8_464": (5.929, 0.068204, 89.396)},
}

# (pp4096, tg128) t/s; a scale variant does not change speed (same kernels)
W4A_SPD = {
    "0.8B": {"w4a4": (31911, 421.7), "w4a8": (28500, 421.6)},
    "27B":  {"w4a4": (2029, 26.68), "w4a8": (1488, 26.66)},
    "35B":  {"w4a4": (4941, 143.9), "w4a8": (3683, 143.8)},
}
W4A_SERIES = ["w4a4_old", "w4a4_uos", "w4a8_256", "w4a8_343", "w4a8_464"]
W4A_STYLE = {"w4a4_old": (PALETTE["other"], None), "w4a4_uos": (PALETTE["other"], "//"),
             "w4a8_256": (PALETTE["mx"], None), "w4a8_343": (PALETTE["mx"], "//"),
             "w4a8_464": ("#A8D44E", None)}
W4A_LABEL = {"w4a4_old": "W4A4 (old scale)", "w4a4_uos": "W4A4 (UOS 7.25)",
             "w4a8_256": "W4A8 (256, shipped)", "w4a8_343": "W4A8 (UOS 343)",
             "w4a8_464": "W4A8 (UOS 464)"}

# ------------------------------------------------------------------ UOS vs e_base KV cache
# Real values, 72-chunk GPU round (mxfp4-imx weights, 2x 5060 Ti, verified vs raw logs):
# (f16-KV control, OCP e_base, UOS 7.25) per model and metric
KVDATA = {
    "0.8B": {"kld": (0.149191, 0.168857, 0.166515), "ppl": (16.1832, 16.4249, 16.3692), "topp": (81.28, 79.99, 80.19)},
    "27B":  {"kld": (0.089920, 0.093152, 0.091820), "ppl": (6.3536, 6.3974, 6.4157), "topp": (90.18, 89.94, 89.93)},
    "35B":  {"kld": (0.068204, 0.073220, 0.072918), "ppl": (5.9285, 5.9521, 5.9504), "topp": (89.40, 88.91, 88.92)},
}


def w4a8_vs_w4a4():
    from matplotlib.ticker import FuncFormatter
    fig, ax = new_figure(2, 3, w=4.6, h=3.6)
    models = list(W4A5.keys())
    acc = [("Mean PPL (lower is better)", 0, [6, 8, 10, 15, 20]),
           ("Mean KLD vs BF16 base (lower is better)", 1, [0.07, 0.1, 0.2, 0.4]),
           ("Same top-p % (higher is better)", 2, [70, 75, 80, 85, 90])]
    for col, (title, idx, yticks) in enumerate(acc):
        style_axes(ax[col], title=title)
        x = np.arange(len(models))
        w = 0.185
        for si, s in enumerate(W4A_SERIES):
            off = (si - 2) * w
            v = [W4A5[m][s][idx] for m in models]
            c, h = W4A_STYLE[s]
            ax[col].bar(x + off, v, width=w, color=c, hatch=h, label=W4A_LABEL[s],
                        edgecolor="white", linewidth=0.4)
        ax[col].set_xticks(x); ax[col].set_xticklabels(models)
        vmin = min(W4A5[m][s][idx] for m in models for s in W4A_SERIES)
        vmax = max(W4A5[m][s][idx] for m in models for s in W4A_SERIES)
        ax[col].set_yscale("log")
        ax[col].set_yticks(yticks)
        ax[col].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax[col].tick_params(which="minor", labelleft=False, labelbottom=False)
        ax[col].set_ylim(vmin * 0.95, vmax * 1.12)
        ax[col].legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2)
    ax[5].set_visible(False)
    spd = [("prefill pp4096 (t/s)", 0, [1000, 2000, 3000, 5000, 10000, 20000, 40000]),
           ("decode tg128 (t/s)", 1, [25, 50, 75, 100, 150, 200, 300, 400, 600])]
    for col, (title, i, yticks) in enumerate(spd):
        style_axes(ax[3 + col], title=title)
        ax[3 + col].set_yscale("log")
        from matplotlib.ticker import FuncFormatter
        ax[3 + col].set_yticks(yticks)
        ax[3 + col].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax[3 + col].tick_params(which="minor", labelleft=False, labelbottom=False)
        x = np.arange(len(models))
        w = 0.38
        v4 = [W4A_SPD[m]["w4a4"][i] for m in models]
        v8 = [W4A_SPD[m]["w4a8"][i] for m in models]
        ax[3 + col].bar(x - w/2, v4, width=w, color=PALETTE["other"], label="W4A4")
        ax[3 + col].bar(x + w/2, v8, width=w, color=PALETTE["mx"], label="W4A8")
        ax[3 + col].set_xticks(x); ax[3 + col].set_xticklabels(models)
        for xi, (a, b) in enumerate(zip(v4, v8)):
            ax[3 + col].text(xi - w/2, a, f"{a:g}", ha="center", va="bottom", fontsize=7, color=PALETTE["muted"])
            ax[3 + col].text(xi + w/2, b, f"{b:g}", ha="center", va="bottom", fontsize=7, color=PALETTE["text"])
        ax[3 + col].set_ylim(min(v4 + v8) * 0.6, max(v4 + v8) * 1.6)
        ax[3 + col].legend(frameon=False, fontsize=7, loc="upper right", ncol=1)
    save(fig, "w4a8-vs-w4a4.png")

def kv_uos():
    from matplotlib.lines import Line2D
    fig, ax = new_figure(3, 3, w=5.0, h=3.2)
    fig.subplots_adjust(hspace=0.55)
    models = list(KVDATA.keys())
    panels = [("Mean KLD vs BF16 base (lower is better)", "kld", "%.6f", 0),
              ("Mean PPL (lower is better)", "ppl", "%.3f", 0.995),
              ("Same top-p % (higher is better)", "topp", "%.2f", 0.98)]
    for row, (title, key, fmt, yfrac) in enumerate(panels):
        for col, m in enumerate(models):
            a = ax[row * 3 + col]
            style_axes(a, title=m, ylabel=title if col == 0 else None)
            c0, e, u = KVDATA[m][key]
            b = a.bar(["OCP e_base", "UOS 7.25"], [e, u],
                      color=[PALETTE["other"], PALETTE["mx"]], width=0.5)
            a.bar_label(b, fmt=fmt, fontsize=7, color=PALETTE["text"], padding=2)
            a.plot([-0.5, 0.5], [c0, c0], color="#000000", ls="--", lw=1.2, alpha=0.3, zorder=2)
            a.text(0.55, c0, f"  f16 {c0:.4f}", fontsize=6.5, color=PALETTE["muted"], va="bottom")
            vmin, vmax = min(c0, e, u), max(c0, e, u)
            a.set_ylim(0 if yfrac == 0 else vmin * yfrac, vmax * 1.3 if yfrac == 0 else vmax + (vmax - vmin) * 0.35)
    save(fig, "kv-uos-vs-ebase.png")



if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "ppl"):
        ppl_vs_size()
    if which in ("all", "kv"):
        kv_cache()
    if which in ("all", "kl"):
        kl_top_p()
    if which in ("all", "kvuos"):
        kv_uos()
    if which in ("all", "w4a8"):
        w4a8_vs_w4a4()
