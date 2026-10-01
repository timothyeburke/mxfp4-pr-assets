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
# fresh full-imatrix, all-chunks, 2026-10-01 refresh (branch 0c666e561).
# (type, size_GB, imx_ppl, no_imx_ppl or None, is_ref); bf16/Q8_0 refs = direct runs
PPL = {
    "0.8B": [
        ("bf16", 1.6, 15.117, None, True),
        ("Q8_0", 0.8, 15.161, None, True),
        ("q5ks", 0.6, 14.917, 15.993, False),
        ("q5_1", 0.6, 15.342, 15.202, False),
        ("q4_1", 0.5, 15.783, 15.800, False),
        ("iq4xs", 0.5, 15.872, 15.970, False),
        ("q4ks", 0.5, 15.993, 15.806, False),
        ("mxfp4", 0.6, 16.179, 17.507, False),
        ("q4_0", 0.5, 18.441, 19.108, False),
        ("q3ks", 0.4, 19.606, 21.511, False),
    ],
    "27B": [
        ("q4ks", 15.6, 6.280, 6.322, False),
        ("q4_1", 17.1, 6.343, 6.333, False),
        ("mxfp4", 15.7, 6.355, 6.449, False),
        ("q5ks", 18.7, 6.370, 6.499, False),
        ("q5_1", 20.3, 6.413, 6.519, False),
        ("iq4xs", 15.1, 6.433, 6.419, False),
        ("bf16", 53.8, 6.424, None, True),
        ("Q8_0", 28.6, 6.439, None, True),
        ("q4_0", 15.5, 6.568, 6.473, False),
        ("q3ks", 12.1, 6.778, 7.373, False),
    ],
    "35B": [
        ("Q8_0", 36.9, 5.742, None, True),
        ("bf16", 69.4, 5.743, None, True),
        ("mxfp4-moe", 19.8, 5.776, 5.805, False),
        ("q5_1", 26.1, 5.759, 5.790, False),
        ("q5ks", 24.0, 5.787, 5.826, False),
        ("q4_1", 21.8, 5.800, 5.887, False),
        ("q4ks", 19.9, 5.806, 5.830, False),
        ("q4_0", 19.8, 5.838, 5.890, False),
        ("iq4xs", 18.7, 5.856, 5.846, False),
        ("mxfp4", 19.0, 5.930, 5.998, False),
        ("q3ks", 15.2, 6.196, 6.531, False),
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
# 2026-10-01 refresh: tg = 2x 5060 Ti (kv-gpu/, q4_1-imx weights), cpu9900 = 9900X tg32
# (kv-cpu2/, mxfp4-imx weights), ppl = 72-chunk GPU (suite C f16/mxfp4 + kv-ppl/)
KV = {
    "0.8B": {"mem": {"f16": 1.14, "q4_0": 0.32, "q4_1": 0.36, "q5_1": 0.43, "mxfp4": 0.30},
             "tg":  {"f16": 440.5, "q4_0": 416.1, "q4_1": 419.4, "q5_1": 419.1, "mxfp4": 418.1},
             "cpu9900": {"f16": 42.6, "q4_0": 44.2, "q4_1": 44.0, "q5_1": 44.0, "mxfp4": 45.8},
             "ppl": {"f16": 16.1791, "q4_0": 16.3715, "q4_1": 16.3278, "q5_1": 16.2197, "mxfp4": 16.3649}},
    "27B":  {"mem": {"f16": 6.10, "q4_0": 1.72, "q4_1": 1.91, "q5_1": 2.29, "mxfp4": 1.62},
             "tg":  {"f16": 42.63, "q4_0": 42.25, "q4_1": 42.27, "q5_1": 42.23, "mxfp4": 42.30},
             "cpu9900": {"f16": 2.13, "q4_0": 2.15, "q4_1": 2.14, "q5_1": 2.17, "mxfp4": 2.15},
             "ppl": {"f16": 6.3552, "q4_0": 6.3749, "q4_1": 6.3689, "q5_1": 6.3578, "mxfp4": 6.4051}},
    "35B":  {"mem": {"f16": 1.91, "q4_0": 0.54, "q4_1": 0.60, "q5_1": 0.72, "mxfp4": 0.51},
             "tg":  {"f16": 184.4, "q4_0": 179.5, "q4_1": 180.2, "q5_1": 180.1, "mxfp4": 180.2},
             "cpu9900": {"f16": 13.3, "q4_0": 12.6, "q4_1": 12.9, "q5_1": 12.8, "mxfp4": 12.8},
             "ppl": {"f16": 5.9302, "q4_0": 5.9442, "q4_1": 5.9415, "q5_1": 5.9330, "mxfp4": 5.9488}},
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
    from matplotlib.ticker import FuncFormatter
    # 4 rows: memory, GPU decode, CPU tg32, PPL effect vs f16 KV (the UOS-vs-e_base
    # KLD story lives in kv-uos-vs-ebase.png)
    fig, ax = new_figure(4, 3, w=5.0, h=3.0)
    for col, (model, d) in enumerate(KV.items()):
        axm = ax[col]
        style_axes(axm, title=model, ylabel="KV mem (GiB @100k)" if col == 0 else None)
        bm = axm.bar(KV_TYPES, [d["mem"][t] for t in KV_TYPES], color=[kv_color(t) for t in KV_TYPES], width=0.72)
        axm.bar_label(bm, fmt="%.2f", fontsize=7, color=PALETTE["text"], padding=2)
        axm.set_ylim(0, max(d["mem"].values()) * 1.22)
        _speed_row(ax[3 + col], d["tg"], "GPU decode (t/s)" if col == 0 else None)
        _speed_row(ax[6 + col], d["cpu9900"], "9900X tg32 (t/s)" if col == 0 else None, log=False)
        # PPL effect vs f16 KV: bar height IS the KV quantization cost
        # PPL effect vs f16 KV: bar height IS the KV quantization cost
        axp = ax[9 + col]
        style_axes(axp, ylabel="PPL effect vs f16 KV" if col == 0 else None)
        f16p = d["ppl"]["f16"]
        kv4 = [t for t in KV_TYPES if t != "f16"]
        eff = [d["ppl"][t] - f16p for t in kv4]
        bp = axp.bar(kv4, eff, color=[kv_color(t) for t in kv4], width=0.6)
        axp.bar_label(bp, fmt="%.4f", fontsize=7, color=PALETTE["text"], padding=2)
        axp.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4f}"))
        vmax = max(eff)
        axp.set_ylim(0, vmax * 1.35)
    save(fig, "kv-cache.png")

# ------------------------------------------------------------------ KL + top-p
# per quant: (imx_kld, imx_topp, plain_kld, plain_topp); q8 has no plain
KL_ORDER = ["q8", "q5_1", "q5ks", "q4_1", "q4ks", "iq4xs", "mxfp4", "mxfp4_moe", "q4_0", "q3ks"]
KL = {
    "0.8B": {
        "q8":    (0.001281, 97.971, None, None),
        "q5_1":  (0.015891, 93.260, 0.025778, 91.695),
        "q5ks":  (0.018831, 92.766, 0.026193, 91.339),
        "q4_1":  (0.052927, 88.291, 0.093340, 84.586),
        "q4ks":  (0.052269, 88.427, 0.076553, 86.354),
        "iq4xs": (0.056050, 88.061, 0.069503, 86.902),
        "mxfp4": (0.149368, 81.280, 0.187334, 78.869),
        "mxfp4_moe": (None, None, None, None),
        "q4_0":  (0.113955, 83.478, 0.143648, 81.411),
        "q3ks":  (0.278545, 74.852, 0.376359, 71.454),
    },
    "27B": {
        "q8":    (0.009265, 98.684, None, None),
        "q5_1":  (0.026894, 96.302, 0.033467, 95.403),
        "q5ks":  (0.026948, 96.090, 0.034250, 95.510),
        "q4_1":  (0.045991, 94.192, 0.067639, 92.145),
        "q4ks":  (0.047592, 93.938, 0.060739, 92.602),
        "iq4xs": (0.039068, 94.321, 0.049152, 93.362),
        "mxfp4": (0.089626, 90.163, 0.100394, 89.265),
        "mxfp4_moe": (None, None, None, None),
        "q4_0":  (0.058135, 92.735, 0.069438, 91.646),
        "q3ks":  (0.114391, 87.856, 0.173964, 85.398),
    },
    "35B": {
        "q8":    (0.005147, 97.397, None, None),
        "q5_1":  (0.013888, 95.255, 0.020513, 94.244),
        "q5ks":  (0.014936, 95.105, 0.020929, 94.245),
        "q4_1":  (0.028568, 93.125, 0.052466, 90.603),
        "q4ks":  (0.029084, 93.111, 0.043588, 91.410),
        "iq4xs": (0.029773, 92.989, 0.038786, 91.922),
        "mxfp4": (0.067925, 89.411, 0.088084, 87.989),
        "mxfp4_moe": (0.028486, 93.433, 0.030573, 93.203),
        "q4_0":  (0.046412, 91.198, 0.056029, 90.334),
        "q3ks":  (0.106165, 86.824, 0.151978, 84.110),
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
# (ppl_w4a4_old, kld_w4a4_old, topp_w4a4_old), w4a4_uos, w4a8_q8, w4a8_256, w4a8_343, w4a8_464
# 2026-10-01 refresh: w4a4_old = master fb27a525d, w4a4_uos = GGML_CUDA_MMQ_PREC=q4,
# w4a8_q8 = GGML_CUDA_MMQ_PREC=q8, 256 = shipped default, 343/464 = patched builds
W4A5 = {
    "0.8B": {"w4a4_old": (21.390, 0.407816, 69.972), "w4a4_uos": (20.055, 0.355445, 71.937),
             "w4a8_q8": (16.084, 0.143213, 81.685), "w4a8_256": (16.179, 0.149368, 81.280),
             "w4a8_343": (16.181, 0.149433, 81.263), "w4a8_464": (16.176, 0.149119, 81.282)},
    "27B":  {"w4a4_old": (6.559, 0.183153, 84.010), "w4a4_uos": (6.506, 0.181412, 84.447),
             "w4a8_q8": (6.345, 0.085965, 90.481), "w4a8_256": (6.355, 0.089626, 90.163),
             "w4a8_343": (6.363, 0.089124, 90.204), "w4a8_464": (6.366, 0.089002, 90.190)},
    "35B":  {"w4a4_old": (6.469, 0.169221, 82.501), "w4a4_uos": (6.344, 0.150270, 83.589),
             "w4a8_q8": (5.913, 0.065651, 89.588), "w4a8_256": (5.930, 0.067925, 89.411),
             "w4a8_343": (5.934, 0.068117, 89.345), "w4a8_464": (5.929, 0.067942, 89.376)},
}

# (pp4096, tg128) t/s; a scale variant does not change speed (same kernels)
# (pp4096, tg128) t/s, 2026-10-01 refresh; scale variants 343/464 share the w4a8 speed
W4A_SPD = {
    "0.8B": {"w4a4": (19997, 421.1), "w4a4_uos": (20220, 425.0), "w4a8": (18719, 424.9), "w4a8_q8": (19933, 424.6)},
    "27B":  {"w4a4": (1869, 46.8), "w4a4_uos": (1872, 46.8), "w4a8": (1475, 46.8), "w4a8_q8": (1542, 46.8)},
    "35B":  {"w4a4": (3868, 178.4), "w4a4_uos": (3865, 179.0), "w4a8": (3032, 179.1), "w4a8_q8": (3631, 179.3)},
}
W4A_SERIES = ["w4a4_old", "w4a4_uos", "w4a8_q8", "w4a8_256", "w4a8_343", "w4a8_464"]
W4A_STYLE = {"w4a4_old": (PALETTE["other"], None), "w4a4_uos": (PALETTE["other"], "//"),
             "w4a8_q8": ("#2E7D32", None),
             "w4a8_256": (PALETTE["mx"], None), "w4a8_343": (PALETTE["mx"], "//"),
             "w4a8_464": ("#A8D44E", None)}
W4A_LABEL = {"w4a4_old": "W4A4 (old scale)", "w4a4_uos": "W4A4 (UOS 7.25)",
             "w4a8_q8": "W4A8-Q8 (Q8_1 acts)",
             "w4a8_256": "W4A8 (256, shipped)", "w4a8_343": "W4A8 (UOS 343)",
             "w4a8_464": "W4A8 (UOS 464)"}
# ------------------------------------------------------------------ UOS vs e_base KV cache
# Real values, 72-chunk GPU round 2026-10-01 (mxfp4-imx weights, 2x 5060 Ti):
# (f16-KV control, OCP e_base, UOS 7.25) per model and metric
KVDATA = {
    "0.8B": {"kld": (0.149368, 0.169209, 0.166633), "ppl": (16.1791, 16.4305, 16.3649), "topp": (81.28, 79.92, 80.15)},
    "27B":  {"kld": (0.089626, 0.092148, 0.093048), "ppl": (6.3552, 6.3971, 6.4051), "topp": (90.16, 89.88, 89.83)},
    "35B":  {"kld": (0.067925, 0.073610, 0.073175), "ppl": (5.9302, 5.9490, 5.9488), "topp": (89.41, 88.94, 88.92)},
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
        x = np.arange(len(models))
        w = 0.15
        for si, s in enumerate(W4A_SERIES):
            off = (si - (len(W4A_SERIES) - 1) / 2) * w
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
        ax[col].legend(frameon=False, fontsize=6, loc="upper right", ncol=2)
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
        x = np.arange(len(models))
        w = 0.21
        arms = ["w4a4", "w4a4_uos", "w4a8_q8", "w4a8"]
        barcols = [PALETTE["other"], PALETTE["other"], "#2E7D32", PALETTE["mx"]]
        hatch = [None, "//", None, None]
        labels = ["W4A4 (old)", "W4A4 (UOS)", "W4A8-Q8", "W4A8"]
        for si, arm in enumerate(arms):
            v = [W4A_SPD[m][arm][i] for m in models]
            ax[3 + col].bar(x + (si - 1.5) * w, v, width=w, color=barcols[si], hatch=hatch[si],
                            edgecolor="white", linewidth=0.4, label=labels[si])
        ax[3 + col].set_xticks(x); ax[3 + col].set_xticklabels(models)
        allv = [W4A_SPD[m][arm][i] for m in models for arm in arms]
        ax[3 + col].set_ylim(min(allv) * 0.6, max(allv) * 1.6)
        ax[3 + col].legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2)
    save(fig, "w4a8-vs-w4a4.png")

def kv_uos():
    # effect = arm - f16-KV control (topp: control - arm), so bar height IS the
    # KV quantization cost (lower is better everywhere on this chart)
    fig, ax = new_figure(3, 3, w=5.0, h=3.2)
    fig.subplots_adjust(hspace=0.55)
    models = list(KVDATA.keys())
    panels = [("KLD effect vs f16 KV", "kld", "%.4f"),
              ("PPL effect vs f16 KV", "ppl", "%.3f"),
              ("top-p loss vs f16 KV", "topp", "%.2f")]
    for row, (title, key, fmt) in enumerate(panels):
        from matplotlib.ticker import FuncFormatter
        for col, m in enumerate(models):
            a = ax[row * 3 + col]
            style_axes(a, title=m, ylabel=title if col == 0 else None)
            c0, e, u = KVDATA[m][key]
            eff = (e - c0, u - c0) if key != "topp" else (c0 - e, c0 - u)
            b = a.bar([0, 1], eff, color=[PALETTE["other"], PALETTE["mx"]], width=0.5)
            a.bar_label(b, fmt=fmt, fontsize=7, color=PALETTE["text"], padding=2)
            a.set_xticks([0, 1]); a.set_xticklabels(["OCP e_base", "UOS 7.25"], fontsize=8)
            a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4f}" if key == "kld" else f"{v:.3f}" if key == "ppl" else f"{v:.2f}"))
            vmax = max(eff)
            a.set_ylim(0, vmax * 1.35)
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
