#!/usr/bin/env python3
"""
mxfp4 imatrix scale-search analysis.

MXFP4 = E2M1 (1 sign, 2 exp, 1 mantissa) in 32-element blocks with a
per-block E8M0 scale. The imatrix-weighted quantizer picks the block scale
by minimizing the imatrix-weighted squared error over a +/-4 window around
the amax-derived base exponent e_base.

This script measures, on a sample of blocks from a BF16 model:
  - how far e_base (and the +/-4 window) is from the weighted GLOBAL
    optimum over all 256 E8M0 exponents,
  - how often the 256-scale error curve has multiple local minima
    (i.e. whether a local search can be trusted),
  - how much weighted error a better search would save.

It replicates the C quantizer exactly (ggml-quants.c):
  - value = kvalues[i] * d, d = 2^(e-128); kvalues = 2 * E2M1, the doubled
    table {0, 1, 2, 3, 4, 6, 8, 12, ...} (E2M1 = {0, .5, 1, 1.5, 2, 3, 4, 6})
  - e_base = lrint(log2(amax) - 2) + 127  (0 if amax == 0)
  - imatrix weight for element j of a tensor = in_sum2[j] / counts[0]
    (per input feature, same for every output row)
  - ties in the nearest-level choice do not change err(e): both equally
    close levels give the same squared error, so a plain argmin is exact.

Outputs (written to --out):
  - curves.npz     per-block err over all 256 exponents + per-block stats
  - summary.json   aggregate statistics
  - per_tensor.csv per-tensor statistics
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "llama.cpp", "gguf-py"))
from gguf import GGUFReader  # noqa: E402

# Doubled E2M1 table (matches kvalues_fp4 in ggml-common.h): value = kvalues[i] * d.
# E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}; doubled -> {0, 1, 2, 3, 4, 6, 8, 12, ...}.
KVALS = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
                 dtype=np.float32)

# E8M0 scale d = 2^(e-128) (ggml_e8m0_to_fp32_half). Finite for all e: e=0 -> 2^-128
# (denormal), e=255 -> 2^127 (max f32). Matches the C quantizer exactly, including
# the 0 * d = 0 behavior at e=255 (no inf/nan).
EXPS = np.arange(256, dtype=np.float32)
with np.errstate(over="ignore", under="ignore", invalid="ignore"):
    D_HALF = np.power(2.0, EXPS - 128.0).astype(np.float32)


def bf16_to_f32(raw: np.ndarray, n: int) -> np.ndarray:
    """Raw uint8 tensor bytes -> float32 (BF16 is the top 16 bits of an f32)."""
    b = np.ascontiguousarray(raw[: n * 2])
    u32 = b.view(np.uint16).astype(np.uint32) << 16
    return u32.view(np.float32)


def lrintf(x: np.ndarray) -> np.ndarray:
    """C lrintf: round half away from zero (vectorized)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def e_base_of(amax: np.ndarray) -> np.ndarray:
    """The no-imatrix scale choice: e = lrint(log2(amax) - 2) + 127."""
    e = np.where(amax > 0.0, lrintf(np.log2(np.maximum(amax, 1e-30)) - 2.0) + 127.0, 0.0)
    return np.clip(e, 0, 255).astype(np.int32)


def err_curve_batch(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted squared error over all 256 exponents for a batch of blocks.

    x, w: shape (B, 32). Returns shape (B, 256).
    value = KVALS[i] * D_HALF[e] (matches the C quantizer exactly).
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        xd = x[:, :, None] / D_HALF[None, None, :]                      # (B, 32, 256)
        diff = np.abs(xd[:, :, :, None] - KVALS[None, None, None, :])  # (B, 32, 256, 16)
        idx = np.argmin(diff, axis=3)                                  # (B, 32, 256)
        q = KVALS[idx] * D_HALF[None, None, :]                        # (B, 32, 256)
        return np.einsum("bj,bje->be", w, (x[:, :, None] - q) ** 2)


def n_local_minima(curves: np.ndarray) -> np.ndarray:
    """Count local minima of each 256-curve (plateau-safe).

    A local min at i: c[i] < c[i-1] and c[i] <= c[i+1] (interior);
    edges count if strictly lower than their only neighbor.
    """
    c = curves
    lt = np.zeros(c.shape, dtype=bool)
    lt[:, 1:] = c[:, 1:] < c[:, :-1]
    lte = np.zeros(c.shape, dtype=bool)
    lte[:, :-1] = c[:, :-1] <= c[:, 1:]
    inner = lt[:, 1:-1] & lte[:, 1:-1]
    left_edge = c[:, 0] < c[:, 1]
    right_edge = c[:, -1] < c[:, -2]
    return inner.sum(axis=1) + left_edge.astype(int) + right_edge.astype(int)


def load_imatrix_weights(im_path: str) -> dict:
    """tensor name -> weight vector (length n_per_row) = in_sum2 / counts[0]."""
    im = GGUFReader(im_path)
    sums, counts = {}, {}
    for t in im.tensors:
        if t.name.endswith(".in_sum2"):
            sums[t.name[: -len(".in_sum2")]] = t.data.astype(np.float32).reshape(-1)
        elif t.name.endswith(".counts"):
            counts[t.name[: -len(".counts")]] = t.data.astype(np.float32).reshape(-1)
    w = {}
    for name, s in sums.items():
        c = counts.get(name)
        if c is None or c.size == 0 or float(c[0]) <= 0.0:
            continue  # mirrors load_imatrix: weight 1 (or skipped)
        w[name] = s / float(c[0])
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/home/tim/.lmstudio/models/ggml-org/"
                    "Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-BF16.gguf")
    ap.add_argument("--imatrix", default="/home/tim/code/services/repos/"
                    "logs/mxfp4-imatrix/imatrix/0.8b.imatrix")
    ap.add_argument("--out", default="/home/tim/code/services/repos/"
                    "logs/mxfp4-scale-search/0.8b")
    ap.add_argument("--per-tensor-frac", type=float, default=1.0 / 200.0,
                    help="fraction of blocks sampled per tensor (min 256)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"loading model: {args.model}")
    model = GGUFReader(args.model)
    weights = load_imatrix_weights(args.imatrix)
    print(f"model tensors: {len(model.tensors)}, imatrix tensors: {len(weights)}")

    # ---- self-test: vectorized err vs naive scalar loop ------------------
    xt = np.float32([0.13, -2.7, 5.9, -0.4, 1.1, 3.3, -4.4, 0.02, 6.1, -1.9,
                    0.7, 2.2, -5.5, 0.3, -0.8, 1.6, 4.2, -3.1, 0.9, -1.2,
                    2.8, -6.3, 0.55, -2.4, 1.35, -0.65, 3.9, -4.9, 0.25,
                    -1.05, 5.05, -0.15])
    wt = np.float32(np.abs(xt) % 1.0 + 0.5)
    naive = []
    for e in range(256):
        d = float(D_HALF[e])
        idx = np.argmin(np.abs(xt[:, None] / d - KVALS[None, :]), axis=1)
        q = KVALS[idx] * d
        naive.append(float(np.dot(wt, (xt - q) ** 2)))
    naive = np.array(naive)
    vec = err_curve_batch(xt[None, :], wt[None, :])[0]
    assert np.allclose(naive, vec, rtol=1e-5), (naive[:8], vec[:8])
    print("self-test OK (vectorized err matches scalar loop)")

    # ---- collect target tensors ------------------------------------------
    targets = []
    for t in model.tensors:
        shape = [int(x) for x in t.shape]
        if len(shape) < 2:
            continue  # 1D (norms) are not quantized
        n_per_row = shape[0]  # ggml ne[0] == first gguf dim (verified vs imatrix size check)
        if n_per_row % 32 != 0:
            continue
        if t.name not in weights:
            print(f"  skip {t.name}: no imatrix entry")
            continue
        targets.append((t, n_per_row, weights[t.name]))
    print(f"target tensors: {len(targets)}")

    # ---- sample blocks + compute -----------------------------------------
    all_rows = []
    per_tensor = []
    n_sampled = 0
    for t, n_per_row, wvec in targets:
        n_el = t.n_elements
        x_all = bf16_to_f32(t.data, n_el)
        n_blocks = n_el // 32
        n_samp = min(n_blocks, max(256, int(n_blocks * args.per_tensor_frac)))
        picks = rng.choice(n_blocks, size=n_samp, replace=False)
        x = x_all.reshape(-1, 32)[picks]
        # imatrix = one per-channel row (length n_per_row), broadcast to every row.
        # C indexes im + (i % nbrow)*qk, nbrow = n_per_row/32 (block pos within row).
        nbrow = n_per_row // 32
        start = (picks % nbrow) * 32
        w = wvec[start[:, None] + np.arange(32)]  # weight per element, per block
        curves = err_curve_batch(x, w)
        amax = np.abs(x).max(axis=1)
        e_base = e_base_of(amax)
        argmin_256 = np.argmin(curves, axis=1)
        lo = np.clip(e_base - 4, 0, 255)
        hi = np.clip(e_base + 4, 0, 255)
        # err restricted to the +/-4 window around e_base (the current quantizer):
        idx256 = np.arange(256)[None, :]
        win_mask = (idx256 >= lo[:, None]) & (idx256 <= hi[:, None])
        win_err = np.where(win_mask, curves, np.inf)
        argmin_pm4 = np.argmin(win_err, axis=1)
        err_ebase = curves[np.arange(n_samp), e_base]
        err_pm4 = curves[np.arange(n_samp), argmin_pm4]
        err_256 = curves.min(axis=1)
        nlm = n_local_minima(curves)

        all_rows.append((curves, e_base, argmin_256, argmin_pm4,
                        err_ebase, err_pm4, err_256, nlm,
                        np.full(n_samp, len(per_tensor), dtype=np.int32), picks,
                        x, wvec))
        n_sampled += n_samp

        # per-tensor stats
        rel_save = (err_ebase - err_256) / np.maximum(err_ebase, 1e-30)
        off = (argmin_256 - e_base).astype(np.int32)
        per_tensor.append({
            "tensor": t.name, "n_per_row": n_per_row, "n_blocks": int(n_blocks),
            "n_sampled": int(n_samp),
            "mean_rel_save_vs_ebase": float(rel_save.mean()),
            "p99_rel_save_vs_ebase": float(np.percentile(rel_save, 99)),
            "frac_window_miss_global": float((np.abs(off) > 4).mean()),
            "mean_abs_offset": float(np.abs(off).mean()),
            "max_abs_offset": int(np.abs(off).max()),
            "mean_n_local_min": float(nlm.mean()),
            "frac_multi_local_min": float((nlm > 1).mean()),
        })
        print(f"  {t.name:32s} blocks={n_blocks:8d} sampled={n_samp:6d} "
              f"off>4: {100*(np.abs(off)>4).mean():5.2f}%  "
              f"save: {100*rel_save.mean():6.2f}%  localmin: {nlm.mean():5.2f}")

    # ---- flatten + save ---------------------------------------------------
    keys = ["curves", "e_base", "argmin_256", "argmin_pm4", "err_ebase",
            "err_pm4", "err_256", "n_local_min", "tensor_id", "block_id"]
    data = {}
    for i, k in enumerate(keys):
        data[k] = np.concatenate([r[i] for r in all_rows])
    data["rel_save_ebase"] = (data["err_ebase"] - data["err_256"]) / np.maximum(data["err_ebase"], 1e-30)
    data["rel_save_pm4"] = (data["err_pm4"] - data["err_256"]) / np.maximum(data["err_ebase"], 1e-30)
    npz_path = os.path.join(args.out, "curves.npz")
    np.savez_compressed(npz_path, **data)
    print(f"wrote {npz_path} ({os.path.getsize(npz_path)/1e6:.1f} MB)")

    rel = data["rel_save_ebase"]
    off = (data["argmin_256"] - data["e_base"]).astype(np.int32)
    summary = {
        "n_blocks_sampled": int(n_sampled),
        "n_tensors": len(per_tensor),
        "mean_rel_save_vs_ebase": float(rel.mean()),
        "p50_rel_save_vs_ebase": float(np.percentile(rel, 50)),
        "p90_rel_save_vs_ebase": float(np.percentile(rel, 90)),
        "p99_rel_save_vs_ebase": float(np.percentile(rel, 99)),
        "frac_window_miss_global": float((np.abs(off) > 4).mean()),
        "frac_offset_1_to_4": float(((np.abs(off) >= 1) & (np.abs(off) <= 4)).mean()),
        "frac_offset_0": float((np.abs(off) == 0).mean()),
        "mean_abs_offset": float(np.abs(off).mean()),
        "p99_abs_offset": float(np.percentile(np.abs(off), 99)),
        "max_abs_offset": int(np.abs(off).max()),
        "mean_n_local_min": float(data["n_local_min"].mean()),
        "frac_multi_local_min": float((data["n_local_min"] > 1).mean()),
        "max_n_local_min": int(data["n_local_min"].max()),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary:\n{json.dumps(summary, indent=2)}")

    with open(os.path.join(args.out, "per_tensor.csv"), "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(per_tensor[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(per_tensor)
    print(f"wrote {os.path.join(args.out, 'per_tensor.csv')}")


if __name__ == "__main__":
    main()
