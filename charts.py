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
        ("mxfp4-moe", 19.8, 5.776, None, False),
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
        ax.annotate(disp, (size, imx), textcoords="offset points", xytext=(7, 0),
                    fontsize=7, color=(PALETTE["mx"] if is_mx else PALETTE["muted"]),
                    ha="left", va="center", zorder=6)
    items = [Line2D([0], [0], marker="o", ls="", ms=9, mec="white", mfc=PALETTE["mx"], label="mxfp4")]
    if any(r[0] == "mxfp4-moe" for r in rows):
        items.append(Line2D([0], [0], marker="D", ls="", ms=9, mec="white", mfc=PALETTE["mx"], label="mxfp4_moe"))
    items.append(Line2D([0], [0], marker="o", ls="", ms=7, mec="white", mfc=PALETTE["other"], label="3/4/5-bit family"))
    items.append(Line2D([0], [0], marker="o", ls="", ms=7, mec=PALETTE["other"], mfc="none", label="no imatrix"))
    items.append(Line2D([0], [0], color=PALETTE["muted"], ls="--", lw=1.2, label="bf16 (ref)"))
    ax.legend(handles=items, loc="upper right", frameon=False, fontsize=8, ncol=1, handlelength=1.4, borderaxespad=0.5)
    return


def ppl_vs_size():
    fig, axgrid = new_figure(1, 3, w=5.4, h=4.3)
    for ax, (model, rows) in zip(axgrid, PPL.items()):
        style_axes(ax, title=model, xlabel="file size (GB)", ylabel="PPL", xlog=True)
        plot_rows = [r for r in rows if r[0] not in ("bf16", "q3ks")]
        scatter_quant_points(ax, plot_rows)
        bf16 = next((r[2] for r in rows if r[0] == "bf16"), None)
        if bf16 is not None:
            ax.axhline(bf16, color=PALETTE["muted"], linestyle="--", linewidth=1.2, zorder=2)
        ax.set_xlim(min(r[1] for r in plot_rows) * 0.98, max(r[1] for r in plot_rows) * 1.06)
        from matplotlib.ticker import FuncFormatter, MaxNLocator
        # log scale, but the range spans < 1 decade: place regular-number ticks with a linear-style locator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}" if v < 10 else f"{v:.0f}"))
        ppls = [r[2] for r in plot_rows]
        ymin, ymax = min(ppls), max(ppls)
        pad = (ymax - ymin) * 0.15
        ax.set_ylim(ymin - pad, ymax)
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
    return PALETTE["mx"] if t == "mxfp4" else PALETTE["other"]

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
        axt.set_yscale("log")
        axt.tick_params(which="minor", labelleft=False, labelbottom=False)
        axt.set_ylim(min(d["tg"].values()) * 0.9, max(d["tg"].values()) * 1.08)
        from matplotlib.ticker import FuncFormatter, MaxNLocator
        # log scale, but the range spans < 1 decade: place regular-number ticks with a linear-style locator
        axt.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        axt.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
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
        "mxfp4_moe": (0.028607, 93.355, None, None),
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
# (ppl_w4a4, ppl_w4a8, pp_w4a4, pp_w4a8, tg_w4a4, tg_w4a8) for the imx file
W4A = {
    "0.8B": (21.390, 16.179, 31911, 28500, 421.7, 421.6),
    "27B":  (6.559, 6.357, 2029, 1488, 26.68, 26.66),
    "35B":  (6.467, 5.933, 4941, 3683, 143.9, 143.8),
}

def w4a8_vs_w4a4():
    fig, ax = new_figure(1, 3, w=4.6, h=3.6)
    panels = [("Mean PPL (lower is better)", 0, 1, False, None),
              ("prefill pp4096 (t/s)", 2, 3, True, [1000, 5000, 10000, 20000, 40000]),
              ("decode tg128 (t/s)", 4, 5, True, [50, 100, 150, 200, 400, 600])]
    models = list(W4A.keys())
    for col, (title, i4, i8, logy, yticks) in enumerate(panels):
        style_axes(ax[col], title=title)
        if logy:
            from matplotlib.ticker import FuncFormatter
            ax[col].set_yscale("log")
            ax[col].set_yticks(yticks)
            ax[col].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
            ax[col].tick_params(which="minor", labelleft=False, labelbottom=False)
        x = np.arange(len(models))
        w = 0.38
        v4 = [W4A[m][i4] for m in models]
        v8 = [W4A[m][i8] for m in models]
        ax[col].bar(x - w/2, v4, width=w, color=PALETTE["other"], label="W4A4 (master)")
        ax[col].bar(x + w/2, v8, width=w, color=PALETTE["mx"], label="W4A8 (branch)")
        ax[col].set_xticks(x); ax[col].set_xticklabels(models)
        for xi, (a, b) in enumerate(zip(v4, v8)):
            ax[col].text(xi - w/2, a, f"{a:g}", ha="center", va="bottom", fontsize=7, color=PALETTE["muted"])
            ax[col].text(xi + w/2, b, f"{b:g}", ha="center", va="bottom", fontsize=7, color=PALETTE["text"])
        if logy:
            ax[col].set_ylim(min(v4 + v8) * 0.6, max(v4 + v8) * 1.6)
        else:
            ax[col].set_ylim(0, max(v4 + v8) * 1.18)
        ax[col].legend(frameon=False, fontsize=7, loc="upper right", ncol=1)
    save(fig, "w4a8-vs-w4a4.png")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "ppl"):
        ppl_vs_size()
    if which in ("all", "kv"):
        kv_cache()
    if which in ("all", "kl"):
        kl_top_p()
    if which in ("all", "w4a8"):
        w4a8_vs_w4a4()
