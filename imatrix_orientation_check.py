#!/usr/bin/env python3
"""Diagnose imatrix orientation vs model tensor shapes (2D and 3D MoE experts).

For each model tensor that has an imatrix entry, report the model shape, the
imatrix (in_sum2) length, and which model dimension the imatrix length equals.
This tells us the correct n_per_row (ggml ne[0]) to use when indexing the
per-input-element imatrix weights.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "llama.cpp", "gguf-py"))
from gguf import GGUFReader  # noqa: E402

MODEL = "/home/tim/.lmstudio/models/ggml-org/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-BF16.gguf"
IM    = "/home/tim/code/services/repos/logs/mxfp4-imatrix/imatrix/35b.imatrix"

model = GGUFReader(MODEL)
im = GGUFReader(IM)
print(f"model tensors: {len(model.tensors)}, imatrix tensors: {len(im.tensors)}")

ims = {}
for t in im.tensors:
    if t.name.endswith(".in_sum2"):
        ims[t.name[: -len(".in_sum2")]] = t.data.astype("float32").reshape(-1)
    elif t.name.endswith(".counts"):
        pass

seen3d = 0
for t in model.tensors:
    if t.name not in ims:
        continue
    shape = [int(x) for x in t.shape]
    L = len(ims[t.name])
    # which dim does L equal?
    matches = [f"dim{i}" for i, d in enumerate(shape) if d == L]
    tag = "3D" if len(shape) == 3 else ("2D" if len(shape) == 2 else f"{len(shape)}D")
    if tag == "3D" and seen3d < 6:
        print(f"\n[{tag}] {t.name:40s} shape={shape}  imatrix_len={L}  matches={matches}")
        seen3d += 1
    elif tag == "2D":
        # only print a few 2D for reference
        pass

# summary: for all, report the fraction matching dim0 vs dim1 vs dim2 (gguf order)
from collections import Counter
c = Counter()
for t in model.tensors:
    if t.name not in ims:
        continue
    shape = [int(x) for x in t.shape]
    L = len(ims[t.name])
    if L in shape:
        # gguf dim index of L (first match)
        idx = shape.index(L)
        c[(len(shape), idx)] += 1
    else:
        c[(len(shape), -1)] += 1  # no dim matches
print("\n=== (ndim, gguf-dim-index-matching-imatrix-len) counts ===")
for k, v in sorted(c.items()):
    print(f"  {k}: {v}")
