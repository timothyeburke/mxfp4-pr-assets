## Overview

End-to-end MXFP4 for CUDA Blackwell, narrowed from #20609: dense MXFP4 ftype, W4A8 block-scaled mma for accuracy, imatrix-driven weight quantization, and an MXFP4 KV cache. Small and mighty at +641/-201 - mostly plumbing to complete the type across ggml quant/dequant, MMQ kernels, FA KV cache, and the llama ftype, plus tests.

- **Dense ftype + KV cache** - `LLAMA_FTYPE_MOSTLY_MXFP4` =42; `-ctk/-ctv mxfp4` KV read directly by the FA vec kernel
- **W4A8 matmul** - activations are intrinsic-quantized on Blackwell to e4m3 and prefill via the block-scaled `mxf8f6f4` mma (e2m1 x e4m3, scale_vec::1X) instead of W4A4 (e2m1 x e2m1). More accurate than the W4A4 path it replaces; the cost is prefill-only, decode is unchanged
- **Scale selection** - measured-optimal e8m0 block scales: used existing fmax=4.0 for e2m1 weights and selected fmax=256 for e4m3 activations, both yielding better perplexity scores than the OCP spec's overflow-safe 6.0 and 448.0 by a wide margin. Optional `--imatrix` weight path picks the optimal per-block weight scale using the importance matrix


## Results

**W4A8 (this PR) vs W4A4 - same mxfp4 files, isolated Blackwell MMA activation change:**

![W4A8 vs W4A4](https://raw.githubusercontent.com/timothyeburke/mxfp4-pr-assets/master/w4a8-vs-w4a4.png)

<details>
<summary>Detail tables</summary>
W4A8 (this PR) vs W4A4 (baseline), same mxfp4 files, 2x 5060 Ti, `-ngl 999 -sm tensor -fa on`. The mxfp4 files load on master with an `unknown type mxfp4` metadata warning and route to the existing W4A4 mma. W4A8 (e4m3 activations) trades a prefill slowdown for a large accuracy win; decode is unchanged.

| model | file | PPL W4A4 | PPL W4A8 | pp4096 W4A4 | pp4096 W4A8 | tg128 W4A4 | tg128 W4A8 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.8B | imx | 21.390 | 16.179 | 31911 | 28500 | 421.7 | 421.6 |
| 0.8B | plain | 23.111 | 17.507 | 31883 | 28513 | 421.7 | 421.8 |
| 27B | imx | 6.559 | 6.357 | 2029 | 1488 | 26.68 | 26.66 |
| 27B | plain | 6.633 | 6.449 | 2025 | 1488 | 26.68 | 26.67 |
| 35B | imx | 6.467 | 5.933 | 4941 | 3683 | 143.9 | 143.8 |
| 35B | plain | 6.514 | 5.998 | 4968 | 3699 | 143.9 | 143.8 |
</details>

**Accuracy vs file size (full imatrix, all chunks):**

![PPL vs file size](https://raw.githubusercontent.com/timothyeburke/mxfp4-pr-assets/master/ppl-vs-size.png)

The 4-bit family clusters tightly in PPL; mxfp4 sits in the pack on the dense models, and on the MoE 35B the MoE recipe (experts mxfp4, dense Q8_0) lands with the 4-bit family while mxfp4-all (dense parts also downgraded to mxfp4) is the outlier.

**KL divergence + same top-p vs BF16 (full imatrix, all chunks):**

![KL divergence + same top-p](https://raw.githubusercontent.com/timothyeburke/mxfp4-pr-assets/master/kl-top-p.png)

KLD and same top-p are both monotonic in bit-width, and the imatrix (filled) beats no-imatrix (hollow) for every quant - it consistently reduces divergence and improves top-p agreement with the bf16 base.

<details>
<summary>Detail tables (PPL + size, full imatrix; pp/tg pending re-measure)</summary>

[Qwen3.8-27B](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) (dense):

| quant | size (GB) | PPL | pp4096 | tg128 |
|---|---:|---:|---:|---:|
| bf16 (ref) | 53.8 | 6.435 | - | - |
| Q8_0 (ref) | 28.6 | 6.434 | - | - |
| q5_1 | 20.3 | 6.417 | - | - |
| q4_1 | 17.1 | 6.342 | - | - |
| iq4xs | 15.1 | 6.438 | - | - |
| q4ks | 15.6 | 6.294 | - | - |
| **mxfp4 (imx)** | 15.7 | **6.364** | - | - |
| q4_0 | 15.5 | 6.572 | - | - |
| q3ks | 12.1 | 6.779 | - | - |

[Qwen3.5-0.8B](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF) (dense):

| quant | size (GB) | PPL | pp4096 | tg128 |
|---|---:|---:|---:|---:|
| bf16 (ref) | 1.6 | 15.129 | - | - |
| Q8_0 (ref) | 0.8 | 15.162 | - | - |
| q5_1 | 0.6 | 15.350 | - | - |
| q4_1 | 0.5 | 15.786 | - | - |
| iq4xs | 0.5 | 15.870 | - | - |
| q4ks | 0.5 | 15.999 | - | - |
| **mxfp4 (imx)** | 0.6 | **16.181** | - | - |
| q4_0 | 0.5 | 18.444 | - | - |
| q3ks | 0.4 | 19.601 | - | - |

[Qwen3.6-35B-A3B](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF) (MoE):

| quant | size (GB) | PPL | pp4096 | tg128 |
|---|---:|---:|---:|---:|
| bf16 (ref) | 69.4 | 5.745 | - | - |
| Q8_0 (ref) | 36.9 | 5.740 | - | - |
| q5_1 | 26.1 | 5.756 | - | - |
| q5ks | 24.0 | 5.785 | - | - |
| q4_1 | 21.8 | 5.804 | - | - |
| q4ks | 19.9 | 5.808 | - | - |
| iq4xs | 18.7 | 5.859 | - | - |
| mxfp4 (all) | 19.0 | 5.934 | - | - |
| **mxfp4 (MoE recipe)** | 19.8 | **5.776** | - | - |
| q4_0 | 19.8 | 5.839 | - | - |
| q3ks | 15.2 | 6.194 | - | - |

pp4096 / tg128 pending re-measurement on the fresh files.

**KL divergence + same top-p vs BF16** (full imatrix, all chunks; imx vs no-imx):

[Qwen3.8-27B](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) (dense):

| quant | KLD (imx) | top-p (imx) | KLD (no-imx) | top-p (no-imx) |
|---|---:|---:|---:|---:|
| q8 (ref) | 0.0091 | 98.67 | - | - |
| q5_1 | 0.0257 | 96.30 | 0.0341 | 95.43 |
| q5ks | 0.0266 | 96.08 | 0.0327 | 95.54 |
| q4_1 | 0.0441 | 94.23 | 0.0686 | 92.06 |
| q4ks | 0.0467 | 94.00 | 0.0599 | 92.59 |
| iq4xs | 0.0396 | 94.32 | 0.0499 | 93.34 |
| **mxfp4** | **0.0898** | **90.22** | 0.1012 | 89.24 |
| q4_0 | 0.0589 | 92.70 | 0.0697 | 91.65 |
| q3ks | 0.1142 | 87.89 | 0.1738 | 85.42 |

[Qwen3.5-0.8B](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF) (dense):

| quant | KLD (imx) | top-p (imx) | KLD (no-imx) | top-p (no-imx) |
|---|---:|---:|---:|---:|
| q8 (ref) | 0.0013 | 98.03 | - | - |
| q5_1 | 0.0159 | 93.28 | 0.0257 | 91.61 |
| q5ks | 0.0187 | 92.76 | 0.0262 | 91.25 |
| q4_1 | 0.0528 | 88.34 | 0.0934 | 84.59 |
| q4ks | 0.0521 | 88.43 | 0.0766 | 86.34 |
| iq4xs | 0.0560 | 88.11 | 0.0694 | 86.88 |
| **mxfp4** | **0.1494** | **81.31** | 0.1872 | 78.83 |
| q4_0 | 0.1138 | 83.58 | 0.1437 | 81.41 |
| q3ks | 0.2785 | 74.85 | 0.3764 | 71.50 |

[Qwen3.6-35B-A3B](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF) (MoE):

| quant | KLD (imx) | top-p (imx) | KLD (no-imx) | top-p (no-imx) |
|---|---:|---:|---:|---:|
| q8 (ref) | 0.0053 | 97.40 | - | - |
| q5_1 | 0.0137 | 95.24 | 0.0204 | 94.31 |
| q5ks | 0.0149 | 95.10 | 0.0216 | 94.10 |
| q4_1 | 0.0291 | 93.07 | 0.0526 | 90.57 |
| q4ks | 0.0291 | 93.07 | 0.0433 | 91.42 |
| iq4xs | 0.0300 | 92.99 | 0.0384 | 91.98 |
| **mxfp4** | **0.0682** | **89.34** | 0.0878 | 87.92 |
| q4_0 | 0.0464 | 91.26 | 0.0559 | 90.31 |
| q3ks | 0.1060 | 86.84 | 0.1515 | 84.14 |
</details>

**KV cache (`-ctk/-ctv mxfp4`):**

![KV cache: memory + decode throughput by KV type](https://raw.githubusercontent.com/timothyeburke/mxfp4-pr-assets/master/kv-cache.png)

**KV cache type is a memory choice, not a speed one:** quantized KV (mxfp4) uses ~3-4x less memory than f16 at long context, while decode throughput is flat across all KV types.

<details>
<summary>Detail tables (KV cache; pp/tg pending re-measure)</summary>

KV cache memory (GiB) at 100k tokens, and GPU throughput (pp4096/tg128, -fa 1), by KV type:

| model | KV | memory @100k | GPU pp4096 | GPU tg128 |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B | f16 | 1.14 GiB | - | - |
|  | q4_0 | 0.32 GiB | - | - |
|  | q4_1 | 0.36 GiB | - | - |
|  | q5_1 | 0.43 GiB | - | - |
|  | **mxfp4** | **0.30 GiB** | - | - |
| Qwen3.8-27B | f16 | 6.10 GiB | - | - |
|  | q4_0 | 1.72 GiB | - | - |
|  | q4_1 | 1.91 GiB | - | - |
|  | q5_1 | 2.29 GiB | - | - |
|  | **mxfp4** | **1.62 GiB** | - | - |
| Qwen3.6-35B-A3B | f16 | 1.91 GiB | - | - |
|  | q4_0 | 0.54 GiB | - | - |
|  | q4_1 | 0.60 GiB | - | - |
|  | q5_1 | 0.72 GiB | - | - |
|  | **mxfp4** | **0.51 GiB** | - | - |

These are hybrid linear/full-attention models - full attention every 4 blocks, so only a fraction of layers grow the KV cache (0.8B: 6 of 24 layers; 27B: 16 of 64; 35B: 10 of 40; KV head dim 256); the sizes above reflect that. GPU pp4096 / tg128 pending re-measurement on the fresh files.

</details>


## Design notes and methodology

<details>
<summary>Why W4A8, scale derivation, and controls</summary>

### Why W4A8 (e4m3 activations)

The previous native MXFP4 path quantized activations to e2m1 (W4A4), the dominant remaining accuracy error source. Measured on the 27B with a 16-chunk harness on wikitext-2 at ctx 4096: e2m1 activations score 6.4944 PPL against 6.2984 for f16 activations. e4m3's 3-bit mantissa and wide exponent range sit close to that bound while keeping the tensor cores. This uses the native mixed-precision `kind::mxf8f6f4` instruction, the sm_120-supported form (see #19662), and is the shared foundation for the MXFP family: mxfp6 and mxfp8 follow-ups use the same plumbing.

Prior art: the closed #27315 improved MXFP4 while keeping e2m1 activations; its own data showed W4A4 at 84.24% same-top-P (KLD 0.1316) vs 89.99% for W4A16 without MMQ. W4A8 takes the near-W4A16 quality with the MMQ speed.

### Scale selection: e8m0 scales stepped in from the format max, plus the optional imatrix weight path

The e8m0 block scale is the standard power-of-two formula, `round_to_pow2(amax / C)`, where C is stepped in from the format's max (the mantissa's largest code), so the block's largest value maps inside the representable range instead of onto its edge. The e2m1 weight uses the codebase's existing C = 4.0 (stepped in from the e2m1 max of 6.0); we found a similar benefit for the e4m3 activation, with a plateau and a cliff past ~320, suggesting the optimum is stepped in from the max for values in range for attention:

- **e2m1 (weight and KV cache): C = 4.0** (the e2m1 max is 6.0) - the codebase's existing value. A flat PPL plateau from ~3-5 with a cliff above 5. 27B: 6.44 at /4.0 vs 6.69 at the spec; 35B: 5.997 vs 6.42.
- **e4m3 (activation): C = 256** (the e4m3 max is 448). A plateau across ~128-256, then a cliff past ~320 (where the grid becomes too coarse to be worth the dynamic range) - the same stepped-in optimum as the weight. 27B: 6.32 at /256 vs 6.69 at the spec's /448.

A single-pass "pick the scale that minimizes output error" (the natural per-layer optimum) is a *worse* default: because the e8m0 scale is a power of two, the per-tensor optimum lands at a different scale than the one that generalizes, and it measured worse (6.5586 vs 6.32 on the 27B). The fixed stepped-in scales are the robust choice.

**Optional imatrix weight quantization.** When calibration data is available, `llama-quantize` can search a band of weight scales around /4.0 and pick, per 32-wide block, the one that minimizes the *imatrix-weighted* output error (weights that the calibration says matter more get their error minimized first). On the 27B this improves on the fixed /4.0 by 0.12 PPL (6.3199 vs 6.4405) - enough to beat every fixed 4-bit quant except Q4_1. It is opt-in (default is the calibration-free /4.0) and is a smaller win on the MoE 35B, where the expert weights dominate and the dense parts that imatrix calibrates are a smaller fraction.

Newly quantized files differ from old ones byte for byte; existing GGUFs are unaffected since dequantization is unchanged.

Prior art: the closed #27315 improved MXFP4 while keeping e2m1 activations (W4A4); this PR's W4A8 + stepped-in-scale + imatrix work takes the near-f16 quality with the MMQ speed.

### Controls and provenance

**Before this PR, MXFP4 was only available for MoE expert weights (MXFP4_MOE). Dense MXFP4 is new here.** All mxfp4 quants are re-quantized from the ggml-org BF16 bases (Qwen3.8-27B-GGUF, Qwen3.6-35B-A3B-GGUF, Qwen3.5-0.8B-GGUF) with this branch's measured-optimal scale (weight /4.0, activation /256) and the full imatrix; the mxfp4 PPL above is the imatrix-weighted variant (27B: 6.364 imx vs 6.441 plain). PPL and KLD/top-p are hardware-independent; the pp/tg bench columns are pending re-measurement on the fresh files.

### How measured

| metric | tool | command flags |
|---|---|---|
| throughput (GPU) | `llama-bench` | `-ngl 999 -sm tensor -fa on -p 4096 -n 128 -r 5` |
| throughput (CPU) | `llama-bench` | `-ngl 0 -t 24 -p 512 -n 32 -r 5` |
| PPL vs BF16 | `llama-perplexity` | `-f wikitext-2 -ngl 999 -sm tensor -fa 1 -c 4096 -b 512` (72 chunks) |
| KLD + same top-p vs BF16 | `llama-perplexity --kl-divergence` | `-f wikitext-2 -c 4096` against a dumped bf16 base (full corpus) |
| imatrix weight quant | `llama-quantize --imatrix` | calibration perplexity run -> importance matrix; per-block weight-scale search around /4.0 |
| weight RMSE | `llama-quantize` / `llama-bench` | vs the dequantized BF16 |

Hardware: 2x RTX 5070 Ti (local) and 2x RTX 5060 Ti @150W (throttled 180W->150W); CPU: Intel Core Ultra 9 285K and AMD Ryzen 9 9900X (24 threads). All models are ggml-org; KLD and PPL are hardware-independent.

</details>

## Scope

- NVFP4/NVFP4_MOE ftypes are intentionally out of scope; #26869 covers the combined NVFP4 quantizer. This PR is MXFP4-only and stays focused on execution, KV, and tests. If maintainers prefer, #26869's author can take the NVFP4 portion and this PR the MXFP4 portion - a clean split along format lines
- mxfp6/mxfp8 dense quants are working in my tree as the next follow-up on the same `mxf8f6f4` plumbing
- Related: #26989 requests the sibling NVFP4 KV cache and #19662 discusses the sm_120 block-scale mma build

## Requirements

- I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES - I used [pi.dev](https://pi.dev) with inference hosted locally using llama.cpp - qwen3.6-27b first, then qwen3.8-27b, on my home GPU servers - for implementation and testing, dogfooding this PR's features throughout: first the mxfp4 KV cache, then an mxfp4-quantized qwen3.8-27b. I own the design, verification, and this description; see `Assisted-by:` in the commit
