# MXFP4 end-to-end for CUDA Blackwell: dense ftype, W4A8 block-scaled mma, MXFP4 KV cache

Dense MXFP4 currently works only as MoE expert weights (MXFP4_MOE). This proposes completing the format end-to-end: a dense ftype, a Blackwell-native MMQ path, and MXFP4 as a KV cache type.

**The whole thing is already built and tested** (+840/-205 across 37 files, including tests and full accuracy/perf data, developed over the past few months, with the earlier exploration in #20609). I'm ready to open PRs for the parts below right away - or, if it's preferred, the complete branch is ready as a single PR.

## What it does

1. **Dense ftype** (`LLAMA_FTYPE_MOSTLY_MXFP4` = 42) + CPU quantizer with optional imatrix per-block weight-scale search (`llama-quantize --imatrix`). Files dequantize through the standard paths on every backend.
1. **Dense ftype** (`LLAMA_FTYPE_MOSTLY_MXFP4` = 42) + CPU quantizer with optional imatrix per-block weight-scale search (`llama-quantize --imatrix`). Files dequantize through the standard paths on every backend.
2. **W4A8 MMQ on Blackwell**: activations quantized to e4m3, prefill via the native block-scaled `mxf8f6f4` mma (the sm_120a-supported form, see #19662; #24364 moved NVFP4 the same W4A8 direction; extends #26675's `ggml_prec` with `GGML_PREC_MXFP8`). W4A4 and Q8_1-activation paths remain selectable via `GGML_CUDA_MMQ_PREC`.
3. **MXFP4 KV cache** (`--cache-type-k/--cache-type-v mxfp4`): smallest quantized KV in the codebase (4.25 bit + e8m0 scale), read directly by the FA vec kernel; the KV scale uses the MXAttention UOS boundary ([arXiv 2607.24377](https://arxiv.org/abs/2607.24377)).
4. **The `mxf8f6f4` mma backbone**: this PR introduces the hardware block-scaled mixed-precision MMA instruction (`mxf8f6f4`: e2m1/e2m3/e3m2/e4m3/e5m2 values x e4m3/e5m2 scales) into the codebase, with the A/B tile loads, fragment layouts and e8m0 scale plumbing shared behind one code path. It buys higher-precision prefill for MXFP4 now, and the same backbone takes MXFP8 (W8A8) and MXFP6 (W6A8) dense weights as direct follow-ups - the type plumbing and quantizer changes those need are the small half, the mma work is already done.

## Top results

Qwen3.5-0.8B / Qwen3.8-27B / Qwen3.6-35B-A3B, 2x RTX 5060 Ti, 72-chunk wikitext-2, KLD vs dumped BF16 bases (full tables and charts in the fork PR):

| metric | W4A4 (#27315) | W4A8 (proposed) | Q8_1 acts (reference) |
|---|---:|---:|---:|
| KLD 27B | 0.183 | 0.090 | 0.086 |
| KLD 35B | 0.169 | 0.068 | 0.066 |
| PPL 27B | 6.559 | 6.355 | 6.345 |
| pp4096 27B | 1869 t/s | 1475 t/s | 1542 t/s |
| tg128 27B | 46.8 t/s | 46.8 t/s | 46.8 t/s |

vs the same files on master, which routes these quants through the W4A4 mma with an "unknown type" warning.

KV cache at 100k context (27B): 6.10 -> 1.62 GiB, PPL penalty +0.050 vs f16 KV (q4_0 KV: +0.020, q5_1: +0.003 - mxfp4 trades a little accuracy for the smallest footprint).

## Proposed split

Each part reviewed and merged independently; 1 and 2 are strict dependencies of 3:

1. `ggml : MXFP4 type + ftype` - type def, CPU quantizer (+imatrix scale search), ftype plumbing, ~12 files (ggml core + gguf-py)
2. `ggml-cuda : MXFP4 MMQ (W4A8 Blackwell)` - mma + MMQ kernels + activation quantization, ~15 files (ggml-cuda)
3. `ggml-cuda : MXFP4 KV cache` - cpy/set-rows/FA vec kernel + cache-type parsing, ~10 files (ggml-cuda)

## Questions

1. Is the 3-part split the right granularity, or is one PR preferred?
2. W4A8 vs W4A4 as the shipped default: W4A4 prefills ~25% faster; W4A8 roughly halves the KLD. The Q8_1 path measures marginally better than both but does not use the format-native instruction. Which way should the default go?
3. KV cache scale: the UOS boundary (Qmax=7.25, data-free, from the MXAttention paper) lowers the KV-cache quantization cost on 0.8B (KLD effect 13% smaller, PPL effect 26% smaller, top-p loss 1.36 -> 1.13) and 35B (KLD 8% smaller); on 27B the whole KV effect is too small to distinguish the formulas. The same UOS boundary also improves the W4A4 activation path where the e2m1 grid is coarse: KLD 12.8%/11.2% lower on 0.8B/35B (27B within noise), PPL 21.39 -> 20.06 (0.8B) and 6.47 -> 6.34 (35B), top-p +2.0/+1.1 pt. Keep UOS as the default for the mxfp4 KV cache (and W4A4 activations)?
