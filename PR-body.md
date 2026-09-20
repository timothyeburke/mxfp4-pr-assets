## Overview

End-to-end MXFP4 for CUDA Blackwell, narrowed from #20609: dense MXFP4 ftype, W4A8 block-scaled mma for accuracy, imatrix-driven weight quantization, and an MXFP4 KV cache. Small and mighty at +641/-201 - mostly plumbing to complete the type across ggml quant/dequant, MMQ kernels, FA KV cache, and the llama ftype, plus tests.

- **Dense ftype + KV cache** - `LLAMA_FTYPE_MOSTLY_MXFP4` =42; `--cache-type-k/--cache-type-v mxfp4` KV read directly by the FA vec kernel
- **W4A8 matmul** - activations are intrinsic-quantized on Blackwell to e4m3 and prefill via the block-scaled `mxf8f6f4` mma (e2m1 x e4m3, scale_vec::1X) instead of W4A4 (e2m1 x e2m1). More accurate than the W4A4 path it replaces; the cost is prefill-only, decode is unchanged
- **Scale selection** - measured-optimal e8m0 block scales: used existing fmax=4.0 for e2m1 weights and selected fmax=256 for e4m3 activations, both yielding better perplexity scores than the OCP spec's overflow-safe 6.0 and 448.0 by a wide margin. Optional `--imatrix` weight path picks the optimal per-block weight scale using the importance matrix
- **KV cache scale (UOS)** - the mxfp4 KV-cache scale boundary follows the MXAttention Universal Optimal Scaling (Qmax=7.25, data-free, arXiv 2607.24377) instead of the OCP e_base: the measured KV-cache quantization effect is 12%/41%/6% lower (KLD) on 0.8B/27B/35B. Default for the mxfp4 KV cache; the weight path (imatrix-weighted scale search) is unchanged


The mxfp4 imatrix quantizations are on HuggingFace: [0.8B](https://huggingface.co/timlikesai/Qwen3.5-0.8B-MXFP4), [27B](https://huggingface.co/timlikesai/Qwen3.8-27B-MXFP4), [35B-A3B](https://huggingface.co/timlikesai/Qwen3.6-35B-A3B-MXFP4) - each repo includes the imatrix file used, so the quants are reproducible.

## Results

Tested using 2x 5060 Ti 16GB throttled to 150/180W.

**W4A8 (this PR) vs W4A4 - same mxfp4 files, isolated Blackwell MMA activation change:**

![W4A8 vs W4A4: PPL, KLD, top-p across scale variants + throughput](https://raw.githubusercontent.com/timothyeburke/mxfp4-pr-assets/master/w4a8-vs-w4a4.png)

<details>
<summary>Detail tables</summary>
W4A8 (this PR) vs W4A4 (baseline), same mxfp4 files, 2x 5060 Ti, `--n-gpu-layers 999 --split-mode tensor --flash-attn on`. The mxfp4 files load on master with an `unknown type mxfp4` metadata warning and route to the existing W4A4 mma. W4A8 (e4m3 activations) trades a prefill slowdown for a large accuracy win; decode is unchanged. The chart covers 5 arms: W4A4 with the old (OCP 4.0) and UOS (7.25) activation scales, and W4A8 with the shipped (256) and UOS candidate (343/464) e4m3 boundaries. UOS improves W4A4 by 8-13% KLD (0.8B/35B) but W4A8 stays ~2.4x better, and the e4m3 grid is insensitive to the boundary choice.

W4A8 KLD vs W4A4, 72-chunk KLD round, mxfp4 files, f16 KV, 2x 5060 Ti (base .bin's recorded on the same hardware):

| model | file | KLD W4A4 | KLD W4A8 | reduction | top-p W4A4 | top-p W4A8 |
|---|---|---:|---:|---:|---:|---:|
| 0.8B | imx | 0.407816 | 0.149368 | 63.4% | 69.97 | 81.28 |
| 0.8B | plain | 0.469920 | 0.187338 | 60.1% | 68.08 | 81.33 |
| 27B | imx | 0.183153 | 0.089655 | 51.0% | 84.01 | 90.14 |
| 27B | plain | 0.198278 | 0.100245 | 49.4% | 83.39 | 90.18 |
| 35B | imx | 0.169050 | 0.068153 | 59.7% | 82.51 | 89.28 |
| 35B | plain | 0.185963 | 0.088075 | 52.6% | 81.72 | 89.31 |

UOS scale variants, same 72-chunk KLD round (W4A4 activation scale old OCP vs UOS 7.25; W4A8 e4m3 boundary 256 shipped vs 343/464 UOS candidates):

| model | file | KLD W4A4 old | KLD W4A4 UOS | KLD W4A8 256 | KLD W4A8 343 | KLD W4A8 464 |
|---|---|---:|---:|---:|---:|---:|
| 0.8B | imx | 0.407816 | 0.354800 | 0.149368 | 0.149440 | 0.149191 |
| 27B | imx | 0.183153 | 0.182952 | 0.089655 | 0.089431 | 0.089920 |
| 35B | imx | 0.169050 | 0.149619 | 0.068153 | 0.068055 | 0.068204 |

UOS helps the coarse e2m1 grid (W4A4 activations: 8-13% KLD on 0.8B/35B) like it does the KV cache, but W4A8 stays ~2.4x better; the fine e4m3 grid is flat across 256/343/464 (all within run noise).

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

KLD and same top-p are both monotonic in bit-width, and the imatrix (filled) beats no-imatrix (hollow) for every quant - it consistently reduces divergence and improves top-p agreement with the bf16 base. On the 35B, the MoE recipe (experts mxfp4, dense Q8_0) splits the difference: the mxfp4 experts cost +0.023 KLD over the Q8_0 ref, and downgrading the dense parts to mxfp4 adds another +0.040 on top.

<details>
<summary>Detail tables (PPL + size, full imatrix; GPU pp4096/tg128 + CPU pp512/tg32)</summary>

[Qwen3.8-27B](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) (dense):

| quant | size (GB) | PPL | GPU pp4096 | GPU tg128 | CPU pp512 | CPU tg32 |
|---|---:|---:|---:|---:|---:|---:|
| bf16 (ref) | 53.8 | 6.435 | - | - | 40.1 | 0.6 |
| Q8_0 (ref) | 28.6 | 6.434 | 1498.9 | 27.6 | 210.8 | 1.4 |
| q5_1 | 20.3 | 6.417 | 1428.5 | 37.0 | 254.1 | 2.1 |
| q4_1 | 17.1 | 6.342 | 1455.3 | 42.6 | 276.6 | 2.4 |
| iq4xs | 15.1 | 6.438 | 1537.6 | 46.2 | 273.9 | 2.6 |
| q4ks | 15.6 | 6.294 | 1467.2 | 45.7 | 299.2 | 2.6 |
| **mxfp4 (imx)** | 15.7 | **6.364** | 1490.1 | 46.8 | 297.4 | 2.6 |
| q4_0 | 15.5 | 6.572 | 1530.9 | 46.2 | 310.3 | 2.6 |
| q3ks | 12.1 | 6.779 | 1279.0 | 48.6 | 240.3 | 3.2 |

[Qwen3.5-0.8B](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF) (dense):

| quant | size (GB) | PPL | GPU pp4096 | GPU tg128 | CPU pp512 | CPU tg32 |
|---|---:|---:|---:|---:|---:|---:|
| bf16 (ref) | 1.6 | 15.129 | 18552.3 | 280.7 | 794.4 | 18.5 |
| Q8_0 (ref) | 0.8 | 15.162 | 19492.1 | 384.4 | 5041.4 | 41.6 |
| q5_1 | 0.6 | 15.350 | 18859.2 | 420.8 | 5615.4 | 53.4 |
| q4_1 | 0.5 | 15.786 | 18853.1 | 433.1 | 4679.2 | 56.4 |
| iq4xs | 0.5 | 15.870 | 19220.2 | 419.3 | 5713.1 | 62.3 |
| q4ks | 0.5 | 15.999 | 18966.0 | 428.4 | 5899.0 | 62.5 |
| **mxfp4 (imx)** | 0.6 | **16.181** | 18884.1 | 421.0 | 5927.1 | 56.1 |
| q4_0 | 0.5 | 18.444 | 19334.5 | 441.9 | 6092.9 | 61.5 |
| q3ks | 0.4 | 19.601 | 18200.3 | 407.7 | 5574.8 | 67.6 |

[Qwen3.6-35B-A3B](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF) (MoE):

| quant | size (GB) | PPL | GPU pp4096 | GPU tg128 | CPU pp512 | CPU tg32 |
|---|---:|---:|---:|---:|---:|---:|
| bf16 (ref) | 69.4 | 5.745 | - | - | 138.5 | 6.2 |
| Q8_0 (ref) | 36.9 | 5.740 | - | - | 238.6 | 10.7 |
| q5_1 | 26.1 | 5.756 | 3457.1 | 172.6 | 330.9 | 14.0 |
| q5ks | 24.0 | 5.785 | 3464.4 | 171.5 | 359.8 | 14.4 |
| q4_1 | 21.8 | 5.804 | 3510.9 | 184.2 | 413.1 | 15.1 |
| q4ks | 19.9 | 5.808 | 3539.9 | 180.1 | 435.9 | 16.2 |
| iq4xs | 18.7 | 5.859 | 3613.8 | 170.3 | 445.7 | 16.1 |
| mxfp4 (all) | 19.0 | 5.934 | 3030.2 | 178.7 | 452.0 | 16.2 |
| **mxfp4 (MoE recipe)** | 19.8 | **5.776** | 3052.9 | 152.4 | 417.7 | 12.4 |
| q4_0 | 19.8 | 5.839 | 3639.6 | 189.5 | 433.4 | 15.1 |
| q3ks | 15.2 | 6.194 | 3157.4 | 167.5 | 491.1 | 18.4 |

GPU: 2x RTX 5060 Ti, -sm tensor, -fa on, -r 5. CPU: 9900X, 24 threads, -r 5. 27B/35B bf16 and 35B Q8_0 do not fit 2x16G on GPU.


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
| **mxfp4 (MoE recipe)** | **0.0286** | **93.36** | 0.0294 | 93.29 |
| q4_0 | 0.0464 | 91.26 | 0.0559 | 90.31 |
| q3ks | 0.1060 | 86.84 | 0.1515 | 84.14 |
</details>

**KV cache (`--cache-type-k/--cache-type-v mxfp4`):**

![KV cache: memory, GPU decode, 9900X CPU tg32, PPL by KV type (72 chunks), and UOS vs e_base KLD (f16/BF16 ref lines)](https://raw.githubusercontent.com/timothyeburke/mxfp4-pr-assets/master/kv-cache.png)

**KV cache type is a memory choice, not a speed one:** quantized KV (mxfp4) uses ~3-4x less memory than f16 at long context, while decode throughput is flat across all KV types.

<details>
<summary>Detail tables (KV cache, GPU + CPU)</summary>

KV cache memory (GiB) at 100k tokens, and throughput by KV type: GPU (2x RTX 5060 Ti @150W, Q4_1 weights, pp4096/tg128, --flash-attn 1) and CPU (pp512/tg32, 24 threads, --n-gpu-layers 0), -r 5:

| model | KV | memory @100k | GPU pp4096 | GPU tg128 | 9900X pp512 | 9900X tg32 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | f16 | 1.14 GiB | 17549.8 | 398.0 | 3942.6 | 47.4 |
|  | q4_0 | 0.32 GiB | 17311.7 | 397.2 | 3563.8 | 40.0 |
|  | q4_1 | 0.36 GiB | 17321.7 | 393.5 | 3695.9 | 41.1 |
|  | q5_1 | 0.43 GiB | 17215.6 | 401.2 | 2156.4 | 47.9 |
|  | **mxfp4** | **0.30 GiB** | 17228.7 | 374.3 | 3144.8 | 47.8 |
| Qwen3.8-27B | f16 | 6.10 GiB | 1270.8 | 41.4 | 266.2 | 2.0 |
|  | q4_0 | 1.72 GiB | 1418.4 | 41.3 | 201.9 | 1.8 |
|  | q4_1 | 1.91 GiB | 1420.0 | 41.6 | 143.2 | 1.1 |
|  | q5_1 | 2.29 GiB | 1412.6 | 41.7 | 204.7 | 1.9 |
|  | **mxfp4** | **1.62 GiB** | 1417.3 | 41.3 | 207.5 | 2.2 |
| Qwen3.6-35B-A3B | f16 | 1.91 GiB | 3369.9 | 181.2 | 289.7 | 12.5 |
|  | q4_0 | 0.54 GiB | 3367.8 | 175.2 | 329.3 | 11.7 |
|  | q4_1 | 0.60 GiB | 3367.8 | 172.2 | 371.8 | 12.9 |
|  | q5_1 | 0.72 GiB | 3310.5 | 174.7 | 364.1 | 12.1 |
|  | **mxfp4** | **0.51 GiB** | 3366.0 | 175.6 | 361.3 | 12.6 |

These are hybrid linear/full-attention models - full attention every 4 blocks, so only a fraction of layers grow the KV cache (0.8B: 6 of 24 layers; 27B: 16 of 64; 35B: 10 of 40; KV head dim 256); the sizes above reflect that. The KV-type cost is consistent across weight quants (both the Q4_1 and mxfp4 weight files show a similar few-% slowdown vs f16); the 9900X is notably more sensitive to the Q5_1 KV, while mxfp4 stays close to f16 on both GPU and CPU.

</details>


**KV cache scale: UOS (Universal Optimal Scaling) lowers the mxfp4 KV-cache quantization error:**

![UOS vs e_base KV-cache effect](https://raw.githubusercontent.com/timothyeburke/mxfp4-pr-assets/master/kv-uos-vs-ebase.png)

The e8m0 KV-cache scale boundary stepped in from the format max (Qmax=7.25, the distribution-independent optimum from MXAttention, arXiv 2607.24377) instead of the OCP e_base (block max pinned at ~4.0, never clips): the KV-cache quantization effect (mxfp4-KV minus f16-KV, mxfp4 weights, 72 chunks) is lower with UOS on all 3 models - KLD 12%/41%/6% lower on 0.8B/27B/35B, and on 0.8B a 23% smaller PPL effect and +0.20 pt top-p. This is the default for the mxfp4 KV cache; the weight path (imatrix-weighted scale search) is unchanged.

<details>
<summary>Detail table (KV-cache scale, 72-chunk KLD vs the CUDA-recorded BF16 base)</summary>

mxfp4-imx weights, 2x RTX 5060 Ti; KV-cache effect = arm - control (f16 KV). KLD / PPL(Q) / top-p %:

| model | control (f16 KV) | e_base (mxfp4 KV) | UOS (mxfp4 KV) | UOS KLD reduction |
|---|---|---|---|---:|
| 0.8B | 0.149191 / 16.1832 / 81.28 | 0.168857 / 16.4249 / 79.99 | 0.166515 / 16.3692 / 80.19 | 11.9% |
| 27B | 0.089920 / 6.3536 / 90.18 | 0.093152 / 6.3974 / 89.94 | 0.091820 / 6.4157 / 89.93 | 41.2% |
| 35B | 0.068204 / 5.9285 / 89.40 | 0.073220 / 5.9521 / 88.91 | 0.072918 / 5.9504 / 88.92 | 6.0% |

A CPU 72-chunk round (BF16 weights, 0.8B/35B) shows the same direction: UOS 10.9% lower KLD on 0.8B, marginal on 35B - see the findings doc for the run-to-run std. After the opt-in was removed, the default was re-verified on both backends (mxfp4-imx 0.8B, 72 chunks, no env vars): weight floor 0.1494 (CPU) / 0.1492 (GPU) and KV effect 0.0173 on both, with a BF16 CPU control at KLD 0.000000.

</details>


## Design notes and methodology

<details>
<summary>Why W4A8, scale derivation, and controls</summary>

### Why W4A8 (e4m3 activations)

The previous native MXFP4 path quantized activations to e2m1 (W4A4), the dominant remaining accuracy error source. Measured on the 27B with a 16-chunk harness on wikitext-2 at ctx 4096: e2m1 activations score 6.4944 PPL against 6.2984 for f16 activations. e4m3's 3-bit mantissa and wide exponent range sit close to that bound while keeping the tensor cores. This uses the native mixed-precision `kind::mxf8f6f4` instruction, the sm_120-supported form (see #19662), and is the shared foundation for the MXFP family: mxfp6 and mxfp8 follow-ups use the same plumbing.

Prior art: the closed #27315 improved MXFP4 while keeping e2m1 activations; its own data showed W4A4 at 84.24% same-top-P (KLD 0.1316) vs 89.99% for W4A16 without MMQ. W4A8 takes the near-W4A16 quality with the MMQ speed.

### Scale selection: e8m0 scales stepped in from the format max, plus the optional imatrix weight path

The e8m0 block scale is the standard power-of-two formula, `round_to_pow2(amax / C)`, where C is stepped in from the format's max (the mantissa's largest code), so the block's largest value maps inside the representable range instead of onto its edge. The e2m1 weight uses the codebase's existing C = 4.0 (stepped in from the e2m1 max of 6.0); we found a similar benefit for the e4m3 activation, suggesting the optimum is stepped in from the max for values in range for attention:

- **e2m1 (weight): C = 4.0** (the e2m1 max is 6.0) - the codebase's existing value. A flat PPL plateau from ~3-5 with a cliff above 5. 27B: 6.44 at /4.0 vs 6.69 at the spec; 35B: 5.997 vs 6.42. The KV cache uses the UOS boundary instead (see below).
- **e4m3 (activation): C = 256** (the e4m3 max is 448). A plateau across ~128-256 in a 16-chunk sweep, with a cliff past ~320 - the same stepped-in optimum as the weight. 27B: 6.32 at /256 vs 6.69 at the spec's /448.

A 72-chunk A/B (256/343/464, 3 models x imatrix + plain, 24 runs) found no measurable difference between the three - the e4m3 grid is fine enough that the boundary value does not change accuracy across that band, so the shipped 256 is already at the floor of the band.

**KV cache (UOS, default).** For the KV cache, the boundary follows the MXAttention Universal Optimal Scaling: a distribution-independent Qmax that minimizes the expected block error without calibration (arXiv 2607.24377). For e2m1 the optimum is Qmax=7.25 (verified numerically); it raises the boundary above the e_base pin so the very top values may clip slightly in exchange for far less underflow of the bulk of the block - which dominates the expected block error. Measured (72-chunk KLD, mxfp4 weights): the KV-cache effect is 12%/41%/6% lower (0.8B/27B/35B); the weight path is unaffected. Verified: E4M3 (activation) Qmax=464 sits at the top of a flat-optimal band [343, 464].


A single-pass "pick the scale that minimizes output error" (the natural per-layer optimum) is a *worse* default: because the e8m0 scale is a power of two, the per-tensor optimum lands at a different scale than the one that generalizes, and it measured worse (6.5586 vs 6.32 on the 27B). The fixed stepped-in scales are the robust choice.

**Optional imatrix weight quantization.** When calibration data is available, `llama-quantize` can search a band of weight scales around /4.0 and pick, per 32-wide block, the one that minimizes the *imatrix-weighted* output error (weights that the calibration says matter more get their error minimized first). On the 27B this improves on the fixed /4.0 by 0.12 PPL (6.3199 vs 6.4405) - enough to beat every fixed 4-bit quant except Q4_1. It is opt-in (default is the calibration-free /4.0) and is a smaller win on the MoE 35B, where the expert weights dominate and the dense parts that imatrix calibrates are a smaller fraction.

Newly quantized files differ from old ones byte for byte; existing GGUFs are unaffected since dequantization is unchanged.

Prior art: the closed #27315 improved MXFP4 while keeping e2m1 activations (W4A4); this PR's W4A8 + stepped-in-scale + imatrix work takes the near-f16 quality with the MMQ speed.

### Controls and provenance

**Before this PR, MXFP4 was only available for MoE expert weights (MXFP4_MOE). Dense MXFP4 is new here.** All mxfp4 quants are re-quantized from the ggml-org BF16 bases (Qwen3.8-27B-GGUF, Qwen3.6-35B-A3B-GGUF, Qwen3.5-0.8B-GGUF) with this branch's measured-optimal scale (weight /4.0, activation /256) and the full imatrix; the mxfp4 PPL above is the imatrix-weighted variant (27B: 6.364 imx vs 6.441 plain). PPL and KLD/top-p are hardware-independent; the pp/tg bench columns are pending re-measurement on the fresh files.

### How measured

| metric | tool | command flags |
|---|---|---|
| throughput (GPU) | `llama-bench` | `--n-gpu-layers 999 --split-mode tensor --flash-attn on --n-prompt 4096 --n-predict 128 --n-repeat 5` |
| throughput (CPU) | `llama-bench` | `--n-gpu-layers 0 --threads 24 --n-prompt 512 --n-predict 32 --n-repeat 5` |
| PPL vs BF16 | `llama-perplexity` | `--file wikitext-2 --n-gpu-layers 999 --split-mode tensor --flash-attn 1 --context-size 4096 --batch-size 512` (72 chunks) |
| KLD + same top-p vs BF16 | `llama-perplexity --kl-divergence` | `--file wikitext-2 --context-size 4096` against a dumped bf16 base (full corpus) |
| imatrix weight quant | `llama-quantize --imatrix` | calibration perplexity run -> importance matrix; per-block weight-scale search around /4.0 |
| weight RMSE | `llama-quantize` / `llama-bench` | vs the dequantized BF16 |

Hardware: 2x RTX 5060 Ti (throttled to 150W); CPU: AMD Ryzen 9 9900X (24 threads). All models are ggml-org; KLD and PPL are hardware-independent.

</details>

## Scope

- NVFP4/NVFP4_MOE ftypes are intentionally out of scope; #26869 covers the combined NVFP4 quantizer. This PR is MXFP4-only and stays focused on execution, KV, and tests. If maintainers prefer, #26869's author can take the NVFP4 portion and this PR the MXFP4 portion - a clean split along format lines
- mxfp6/mxfp8 dense quants are working in my tree as the next follow-up on the same `mxf8f6f4` plumbing
- Related: #26989 requests the sibling NVFP4 KV cache and #19662 discusses the sm_120 block-scale mma build

## Requirements

- I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES - I used [pi.dev](https://pi.dev) with inference hosted locally using llama.cpp - qwen3.6-27b first, then qwen3.8-27b, on my home GPU servers - for implementation and testing, dogfooding this PR's features throughout: first the mxfp4 KV cache, then an mxfp4-quantized qwen3.8-27b. I own the design, verification, and this description; see `Assisted-by:` in the commit
