## Overview

End-to-end MXFP4 for CUDA Blackwell, narrowed from #20609: dense MXFP4 ftype, W4A8 block-scaled mma, imatrix-driven weight quantization, and an MXFP4 KV cache. Small and mighty at +840/-205 - mostly plumbing to complete the type across ggml quant/dequant, MMQ kernels, FA KV cache, and the llama ftype, plus tests.

- **Dense ftype + KV cache** - `LLAMA_FTYPE_MOSTLY_MXFP4` =42; `--cache-type-k/--cache-type-v mxfp4` KV read directly by the FA vec kernel
- **W4A8 matmul** - activations are intrinsic-quantized on Blackwell to e4m3 and prefill via the native block-scaled W4A8 `mxf8f6f4` mma instead of W4A4: ~2.4x lower KLD at unchanged decode. The same instruction and mostly the same code back the mxfp8/mxfp6 (W8A8/W6A8) weight follow-ups; the standard Q8_1-activation MMQ path remains selectable (`GGML_CUDA_MMQ_PREC=q8`)
- **Scale selection** - existing fmax=4.0 for e2m1 weights and fmax=256 for e4m3 activations, both yielding better perplexity scores than the OCP spec's overflow-safe 6.0 and 448.0. Optional `--imatrix` weight path picks the optimal per-block weight scale using the importance matrix
- **KV cache scale (UOS)** - the mxfp4 KV-cache scale boundary follows the MXAttention Universal Optimal Scaling ([arXiv 2607.24377](https://arxiv.org/abs/2607.24377)): the measured KV-cache quantization effect is lower (KLD) on 0.8B/35B, within run noise on 27B. Default for mxfp4 KV cache; the weight path is unchanged
- **KV cache scale (UOS)** - the mxfp4 KV-cache scale boundary follows the MXAttention Universal Optimal Scaling ([arXiv 2607.24377](https://arxiv.org/abs/2607.24377)): the measured KV-cache quantization effect is lower (KLD) on 0.8B, within run noise on 27B/35B. Default for mxfp4 KV cache; the weight path is unchanged

The mxfp4 imatrix quantizations are on HuggingFace: [0.8B](https://huggingface.co/timlikesai/Qwen3.5-0.8B-MXFP4), [27B](https://huggingface.co/timlikesai/Qwen3.8-27B-MXFP4), [35B-A3B](https://huggingface.co/timlikesai/Qwen3.6-35B-A3B-MXFP4) - each repo includes the imatrix file used, so the quants are reproducible.

## Results

Tested using 2x 5060 Ti 16GB throttled to 150W/180W due to a slightly defective power supply.

### W4A8 (this PR) vs W4A4 vs Q8 activations

![W4A8 vs W4A4 vs Q8 activations: PPL, KLD, top-p across scale variants + throughput](https://raw.githubusercontent.com/timothyeburke/mxfp4-pr-assets/master/w4a8-vs-w4a4.png)

<details>
<summary>Details</summary>

Same mxfp4 files, 2x 5060 Ti, `--n-gpu-layers 999 --split-mode tensor --flash-attn on`. Four MMQ activation arms: W4A4 with the old OCP scale (master `fb27a525d`, which loads these files with an `unknown type mxfp4` warning and routes to its existing W4A4 mma), W4A4 with the UOS e2m1 activation scale (this branch, `GGML_CUDA_MMQ_PREC=q4`), the shipped W4A8 `mxf8f6f4` path (e4m3 activations, default), and the standard Q8_1-activation MMQ path (`GGML_CUDA_MMQ_PREC=q8`). W4A8 cuts W4A4's KLD by ~2.4-2.7x; decode is unchanged across arms.

W4A arms, imx (imatrix) files, 72-chunk KLD, f16 KV, vs the dumped BF16 base:

| model | arm | PPL | KLD | top-p |
|---|---|---:|---:|---:|
| 0.8B | W4A4 (old scale) | 21.390 | 0.407816 | 69.97 |
| 0.8B | W4A4 (UOS 7.25) | 20.055 | 0.355445 | 71.94 |
| 0.8B | W4A8 (256, shipped) | 16.179 | 0.149368 | 81.28 |
| 0.8B | W4A8-Q8 (Q8_1 acts) | 16.084 | 0.143213 | 81.69 |
| 27B | W4A4 (old scale) | 6.559 | 0.183153 | 84.01 |
| 27B | W4A4 (UOS 7.25) | 6.506 | 0.181412 | 84.45 |
| 27B | W4A8 (256, shipped) | 6.355 | 0.089626 | 90.16 |
| 27B | W4A8-Q8 (Q8_1 acts) | 6.345 | 0.085965 | 90.48 |
| 35B | W4A4 (old scale) | 6.469 | 0.169221 | 82.50 |
| 35B | W4A4 (UOS 7.25) | 6.344 | 0.150270 | 83.59 |
| 35B | W4A8 (256, shipped) | 5.930 | 0.067925 | 89.41 |
| 35B | W4A8-Q8 (Q8_1 acts) | 5.913 | 0.065651 | 89.59 |

Same for the plain (no-imatrix) files:

| model | arm | PPL | KLD | top-p |
|---|---|---:|---:|---:|
| 0.8B | W4A4 (old scale) | 23.111 | 0.469920 | 68.08 |
| 0.8B | W4A4 (UOS 7.25) | 21.787 | 0.412132 | 69.85 |
| 0.8B | W4A8 (256, shipped) | 17.507 | 0.187334 | 78.87 |
| 0.8B | W4A8-Q8 (Q8_1 acts) | 17.370 | 0.180401 | 79.25 |
| 27B | W4A4 (old scale) | 6.633 | 0.198278 | 83.39 |
| 27B | W4A4 (UOS 7.25) | 6.604 | 0.191065 | 83.93 |
| 27B | W4A8 (256, shipped) | 6.449 | 0.100394 | 89.27 |
| 27B | W4A8-Q8 (Q8_1 acts) | 6.425 | 0.098277 | 89.50 |
| 35B | W4A4 (old scale) | 6.514 | 0.185963 | 81.72 |
| 35B | W4A4 (UOS 7.25) | 6.413 | 0.169811 | 82.83 |
| 35B | W4A8 (256, shipped) | 5.998 | 0.088084 | 87.99 |
| 35B | W4A8-Q8 (Q8_1 acts) | 5.978 | 0.085136 | 88.13 |

UOS scale variants, same 72-chunk KLD (W4A4 activation scale old OCP vs UOS 7.25; W4A8 e4m3 boundary 256 shipped vs 343/464 UOS candidates), imx files:

| model | KLD W4A4 old | KLD W4A4 UOS | KLD W4A8 256 | KLD W4A8 343 | KLD W4A8 464 |
|---|---:|---:|---:|---:|---:|
| 0.8B | 0.407816 | 0.355445 | 0.149368 | 0.149433 | 0.149119 |
| 27B | 0.183153 | 0.181412 | 0.089626 | 0.089124 | 0.089002 |
| 35B | 0.169221 | 0.150270 | 0.067925 | 0.068117 | 0.067942 |

UOS helps the coarse e2m1 grid (W4A4 activations: 8-13% KLD on 0.8B/35B, neutral on 27B) like it does the KV cache, but W4A8 stays ~2.4x better; the fine e4m3 grid is flat across 256/343/464 (all within run noise). The Q8_1-activation path measures slightly better than W4A8 on both accuracy and prefill on this hardware.

Throughput (imx files, pp4096 / tg128, -r 5):

| model | W4A4 old | W4A4 UOS | W4A8 | W4A8-Q8 |
|---|---:|---:|---:|---:|
| 0.8B | 19997 / 421.1 | 20220 / 425.0 | 18719 / 424.9 | 19933 / 424.6 |
| 27B | 1869 / 46.8 | 1872 / 46.8 | 1475 / 46.8 | 1542 / 46.8 |
| 35B | 3868 / 178.4 | 3865 / 179.0 | 3032 / 179.1 | 3631 / 179.3 |

</details>

### Accuracy vs file size

![PPL vs file size](https://raw.githubusercontent.com/timothyeburke/mxfp4-pr-assets/master/ppl-vs-size.png)
<details>
<summary>Details</summary>

[Qwen3.8-27B](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) (dense):

| quant | size (GB) | PPL | GPU pp4096 | GPU tg128 | CPU pp512 | CPU tg32 |
|---|---:|---:|---:|---:|---:|---:|
| bf16 (ref) | 53.8 | 6.424 | - | - | 108.6 | 0.7 |
| Q8_0 (ref) | 28.6 | 6.439 | 1518.6 | 27.5 | 156.3 | 1.2 |
| q4ks | 15.6 | 6.280 | 1488.3 | 45.8 | 220.2 | 2.0 |
| q4_1 | 17.1 | 6.343 | 1473.9 | 42.6 | 221.3 | 1.9 |
| **mxfp4 (imx)** | 15.7 | **6.355** | 1476.0 | 46.4 | 225.4 | 2.1 |
| q5ks | 18.7 | 6.370 | 1445.3 | 39.4 | 212.1 | 1.8 |
| q5_1 | 20.3 | 6.413 | 1448.0 | 37.0 | 203.5 | 1.7 |
| iq4xs | 15.1 | 6.433 | 1557.7 | 46.7 | 190.8 | 2.1 |
| q4_0 | 15.5 | 6.568 | 1558.0 | 46.2 | 241.4 | 2.1 |
| q3ks | 12.1 | 6.778 | 1321.4 | 52.2 | 242.6 | 2.5 |

[Qwen3.5-0.8B](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF) (dense):

| quant | size (GB) | PPL | GPU pp4096 | GPU tg128 | CPU pp512 | CPU tg32 |
|---|---:|---:|---:|---:|---:|---:|
| bf16 (ref) | 1.6 | 15.117 | 19649.4 | 281.6 | 769.9 | 18.6 |
| Q8_0 (ref) | 0.8 | 15.161 | 19682.1 | 388.5 | 2613.4 | 31.4 |
| q5ks | 0.6 | 14.917 | 18953.4 | 422.7 | 2120.6 | 44.2 |
| q5_1 | 0.6 | 15.342 | 19022.8 | 427.0 | 1463.1 | 40.2 |
| q4_1 | 0.5 | 15.783 | 19061.7 | 440.1 | 2043.2 | 46.2 |
| iq4xs | 0.5 | 15.872 | 19275.9 | 423.3 | 1931.4 | 50.4 |
| q4ks | 0.5 | 15.993 | 19081.0 | 434.1 | 3304.2 | 48.7 |
| **mxfp4 (imx)** | 0.6 | **16.179** | 18520.3 | 423.8 | 2489.4 | 48.6 |
| q4_0 | 0.5 | 18.441 | 19470.6 | 445.8 | 1916.4 | 52.0 |
| q3ks | 0.4 | 19.606 | 18371.6 | 415.9 | 1685.8 | 59.6 |

[Qwen3.6-35B-A3B](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF) (MoE):

| quant | size (GB) | PPL | GPU pp4096 | GPU tg128 | CPU pp512 | CPU tg32 |
|---|---:|---:|---:|---:|---:|---:|
| bf16 (ref) | 69.4 | 5.743 | - | - | 125.0 | 5.0 |
| Q8_0 (ref) | 36.9 | 5.742 | - | - | 215.8 | 8.1 |
| q5_1 | 26.1 | 5.759 | 3490.3 | 173.1 | 273.4 | 10.8 |
| q5ks | 24.0 | 5.787 | 3503.3 | 172.2 | 317.6 | 11.3 |
| q4_1 | 21.8 | 5.800 | 3541.7 | 184.4 | 344.6 | 11.4 |
| q4ks | 19.9 | 5.806 | 3567.2 | 181.0 | 322.9 | 13.6 |
| q4_0 | 19.8 | 5.838 | 3670.5 | 190.0 | 327.8 | 12.9 |
| iq4xs | 18.7 | 5.856 | 3666.7 | 170.7 | 295.7 | 12.0 |
| mxfp4 (all) | 19.0 | 5.930 | 3040.0 | 179.4 | 377.6 | 13.0 |
| **mxfp4 (MoE recipe)** | 19.8 | **5.776** | 3084.2 | 152.8 | 337.6 | 10.1 |
| q3ks | 15.2 | 6.196 | 3231.1 | 174.9 | 395.2 | 14.8 |
GPU: 2x RTX 5060 Ti, -sm tensor, -fa on, -r 5. CPU: 9900X, 24 threads, -fa on, -r 5. 27B/35B bf16 and 35B Q8_0 do not fit 2x16G on GPU.
</details>

### KL divergence + same top-p vs BF16

![KL divergence + same top-p](https://raw.githubusercontent.com/timothyeburke/mxfp4-pr-assets/master/kl-top-p.png)

<details>
<summary>Details</summary>

[Qwen3.8-27B](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) (dense):

| quant | KLD (imx) | top-p (imx) | KLD (no-imx) | top-p (no-imx) |
|---|---:|---:|---:|---:|
| q8 (ref) | 0.0093 | 98.68 | - | - |
| q5_1 | 0.0269 | 96.30 | 0.0335 | 95.40 |
| q5ks | 0.0269 | 96.09 | 0.0343 | 95.51 |
| q4_1 | 0.0460 | 94.19 | 0.0676 | 92.14 |
| q4ks | 0.0476 | 93.94 | 0.0607 | 92.60 |
| iq4xs | 0.0391 | 94.32 | 0.0492 | 93.36 |
| **mxfp4** | **0.0896** | **90.16** | 0.1004 | 89.27 |
| q4_0 | 0.0581 | 92.73 | 0.0694 | 91.65 |
| q3ks | 0.1144 | 87.86 | 0.1740 | 85.40 |

[Qwen3.5-0.8B](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF) (dense):

| quant | KLD (imx) | top-p (imx) | KLD (no-imx) | top-p (no-imx) |
|---|---:|---:|---:|---:|
| q8 (ref) | 0.0013 | 97.97 | - | - |
| q5_1 | 0.0159 | 93.26 | 0.0258 | 91.69 |
| q5ks | 0.0188 | 92.77 | 0.0262 | 91.34 |
| q4_1 | 0.0529 | 88.29 | 0.0933 | 84.59 |
| q4ks | 0.0523 | 88.43 | 0.0766 | 86.35 |
| iq4xs | 0.0561 | 88.06 | 0.0695 | 86.90 |
| **mxfp4** | **0.1494** | **81.28** | 0.1873 | 78.87 |
| q4_0 | 0.1140 | 83.48 | 0.1436 | 81.41 |
| q3ks | 0.2785 | 74.85 | 0.3764 | 71.45 |

[Qwen3.6-35B-A3B](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF) (MoE):

| quant | KLD (imx) | top-p (imx) | KLD (no-imx) | top-p (no-imx) |
|---|---:|---:|---:|---:|
| q8 (ref) | 0.0051 | 97.40 | - | - |
| q5_1 | 0.0139 | 95.25 | 0.0205 | 94.24 |
| q5ks | 0.0149 | 95.11 | 0.0209 | 94.25 |
| q4_1 | 0.0286 | 93.12 | 0.0525 | 90.60 |
| q4ks | 0.0291 | 93.11 | 0.0436 | 91.41 |
| iq4xs | 0.0298 | 92.99 | 0.0388 | 91.92 |
| **mxfp4** | **0.0679** | **89.41** | 0.0881 | 87.99 |
| **mxfp4 (MoE recipe)** | **0.0285** | **93.43** | 0.0306 | 93.20 |
| q4_0 | 0.0464 | 91.20 | 0.0560 | 90.33 |
| q3ks | 0.1062 | 86.82 | 0.1520 | 84.11 |
</details>

### KV cache (`--cache-type-k/--cache-type-v mxfp4`)

![KV cache: memory, GPU decode, 9900X CPU tg32, PPL by KV type (72 chunks), and UOS vs e_base KLD (f16/BF16 ref lines)](https://raw.githubusercontent.com/timothyeburke/mxfp4-pr-assets/master/kv-cache.png)

<details>
<summary>Details</summary>

KV cache memory (GiB) at 100k tokens, and throughput by KV type: GPU (2x RTX 5060 Ti, 150W, Q4_1-imx weights, pp4096/tg128, --flash-attn 1) and CPU (9900X, pp512/tg32, 24 threads, --n-gpu-layers 0, MXFP4-imx weights), -r 5:

| model | KV | memory (100k) | GPU pp4096 | GPU tg128 | 9900X pp512 | 9900X tg32 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | f16 | 1.14 GiB | 19258.4 | 440.5 | 4149.2 | 42.6 |
|  | q4_0 | 0.32 GiB | 19014.1 | 416.1 | 2839.7 | 44.2 |
|  | q4_1 | 0.36 GiB | 19039.9 | 419.4 | 3090.6 | 44.0 |
|  | q5_1 | 0.43 GiB | 18957.8 | 419.1 | 3032.5 | 44.0 |
|  | **mxfp4** | **0.30 GiB** | 18958.6 | 418.1 | 3142.7 | 45.8 |
| Qwen3.8-27B | f16 | 6.10 GiB | 1472.3 | 42.6 | 208.3 | 2.1 |
|  | q4_0 | 1.72 GiB | 1468.6 | 42.3 | 208.0 | 2.2 |
|  | q4_1 | 1.91 GiB | 1466.7 | 42.3 | 239.3 | 2.1 |
|  | q5_1 | 2.29 GiB | 1466.4 | 42.2 | 234.5 | 2.2 |
|  | **mxfp4** | **1.62 GiB** | 1466.3 | 42.3 | 230.1 | 2.2 |
| Qwen3.6-35B-A3B | f16 | 1.91 GiB | 3542.9 | 184.4 | 380.6 | 13.3 |
|  | q4_0 | 0.54 GiB | 3529.1 | 179.5 | 363.5 | 12.6 |
|  | q4_1 | 0.60 GiB | 3526.7 | 180.2 | 349.7 | 12.9 |
|  | q5_1 | 0.72 GiB | 3523.0 | 180.1 | 348.8 | 12.8 |
|  | **mxfp4** | **0.51 GiB** | 3523.2 | 180.2 | 373.7 | 12.8 |

72-chunk PPL by KV type (mxfp4-imx weights, GPU): 0.8B f16 16.1791 / q5_1 16.2197 / q4_1 16.3278 / mxfp4 16.3649 / q4_0 16.3715; 27B f16 6.3552 / q5_1 6.3578 / q4_1 6.3689 / q4_0 6.3749 / mxfp4 6.4051; 35B f16 5.9302 / q5_1 5.9330 / q4_1 5.9415 / q4_0 5.9442 / mxfp4 5.9488.

</details>

### KV cache scale
UOS (Universal Optimal Scaling) lowers the mxfp4 KV-cache quantization error on 0.8B; on 27B/35B the two arms are within run noise.

![UOS vs e_base KV-cache effect](https://raw.githubusercontent.com/timothyeburke/mxfp4-pr-assets/master/kv-uos-vs-ebase.png)

<details>
<summary>Details</summary>

mxfp4-imx weights, 2x RTX 5060 Ti; KV-cache effect = arm - control (f16 KV). KLD / PPL(Q) / top-p %:

| model | control (f16 KV) | e_base (mxfp4 KV) | UOS (mxfp4 KV) | UOS KLD effect |
|---|---|---|---|---:|
| 0.8B | 0.149368 / 16.1791 / 81.28 | 0.169209 / 16.4305 / 79.92 | 0.166633 / 16.3649 / 80.15 | 13.0% lower |
| 35B | 0.067925 / 5.9302 / 89.41 | 0.073610 / 5.9490 / 88.94 | 0.073175 / 5.9488 / 88.92 | within noise |

The 27B KV effect itself is only +0.0025 to +0.0034 KLD over the f16 control - below the per-run std (+/-0.0019) - so neither scale formula is distinguishable there. UOS is kept as the default on the strength of the 0.8B result (and the W4A4-activation result, where UOS is also never worse).
</details>


## Design notes and methodology

<details>
<summary>Why W4A8, scale derivation, and controls</summary>

### Why W4A8

The previous native MXFP4 path quantized activations to e2m1 (W4A4). W4A8 uses e4m3 activations via the native mixed-precision `kind::mxf8f6f4` mma - the sm_120-supported form (see #19662) - cutting W4A4's KLD by ~2.4x at unchanged decode; the same instruction and mostly the same code back the mxfp8/mxfp6 (W8A8/W6A8) weight follow-ups. The standard Q8_1-activation MMQ path (`GGML_CUDA_MMQ_PREC=q8`) stays selectable and measures slightly better than W4A8 on both accuracy and prefill on this hardware (Q8_1 activations carry more precision than e4m3); W4A8 remains the default as the format-native instruction path. Prior art: the closed #27315 kept e2m1 activations; its data showed W4A4 at 84.24% same-top-P (KLD 0.1316) vs 89.99% for W4A16 without MMQ.

### Scale selection

The e8m0 block scale is the standard `round_to_pow2(amax / C)` with C stepped in from the format's max, so the block's largest value maps inside the representable range instead of onto its edge:
- **e2m1 (weight): C = 4.0** (the e2m1 max is 6.0) - the codebase's existing value
- **e4m3 (activation): C = 256** (the e4m3 max is 448) - selected by sweep; a 72-chunk A/B found the e4m3 grid insensitive to the boundary (see the W4A8 section)
- **KV cache (UOS, default):** the MXAttention Universal Optimal Scaling boundary, Qmax=7.25, data-free (arXiv 2607.24377) - see the KV-cache scale section
- **KV cache (UOS, default):** the MXAttention Universal Optimal Scaling boundary, Qmax=7.25, data-free (arXiv 2607.24377) - see the KV-cache scale section
- **W4A4 activations (opt-in, `GGML_CUDA_MMQ_PREC=q4`): UOS 7.25** - the same MXAttention boundary for the e2m1 activation grid. Improves KLD/PPL vs the OCP 4.0 pin (0.8B KLD 0.408 -> 0.355, PPL 21.39 -> 20.06; 35B KLD 0.169 -> 0.150, PPL 6.47 -> 6.34; 27B within noise) - see the W4A8 section
A single-pass "pick the scale that minimizes output error" (the natural per-tensor optimum) measured worse than the fixed stepped-in scales on an early sweep; the e8m0 scale is a power of two, so the per-tensor optimum does not generalize. The fixed stepped-in scales are the robust choice.

**Optional imatrix weight quantization.** With calibration data, `llama-quantize --imatrix` searches a band of weight scales around /4.0 and picks, per 32-wide block, the one that minimizes the imatrix-weighted output error. Opt-in; the default is the calibration-free /4.0.

Existing GGUFs are unaffected; dequantization is unchanged.


### Controls

**Before this PR, MXFP4 was only available for MoE expert weights (MXFP4_MOE). Dense MXFP4 is new here.** All mxfp4 quants are quantized from the ggml-org BF16 bases (Qwen3.8-27B-GGUF, Qwen3.6-35B-A3B-GGUF, Qwen3.5-0.8B-GGUF) with the measured-optimal scales (weight /4.0, activation /256); the mxfp4 PPL above is the imatrix-weighted variant (27B: 6.355 imx vs 6.449 plain). Results are from this branch (0c666e561); the W4A4-old arm is current master (fb27a525d).

### How measured

| metric | tool | command flags |
|---|---|---|
| throughput (GPU) | `llama-bench` | `--n-gpu-layers 999 --split-mode tensor --flash-attn on --n-prompt 4096 --n-predict 128 --n-repeat 5` |
| throughput (CPU) | `llama-bench` | `--n-gpu-layers 0 --threads 24 --n-prompt 512 --n-predict 32 --n-repeat 5` |
| PPL vs BF16 | `llama-perplexity` | `--file wikitext-2 --n-gpu-layers 999 --split-mode tensor --flash-attn 1 --context-size 4096 --batch-size 512` (72 chunks) |
| KLD + same top-p vs BF16 | `llama-perplexity --kl-divergence` | `--file wikitext-2 --context-size 4096` against a dumped bf16 base (full corpus) |
| W4A activation arms | env | default = W4A8 `mxf8f6f4`; `GGML_CUDA_MMQ_PREC=q4` = W4A4 (UOS e2m1 scale); `GGML_CUDA_MMQ_PREC=q8` = standard Q8_1 activations; W4A4-old = master build |
| imatrix weight quant | `llama-quantize --imatrix` | calibration perplexity run -> importance matrix; per-block weight-scale search around /4.0 |
| weight RMSE | `llama-quantize` / `llama-bench` | vs the dequantized BF16 |

Hardware: 2x RTX 5060 Ti (throttled to 150W); CPU: AMD Ryzen 9 9900X (24 threads). All models are ggml-org; KLD and PPL are hardware-independent.

</details>

## Scope

- NVFP4/NVFP4_MOE ftypes are intentionally out of scope; #26869 covers the combined NVFP4 quantizer. This PR is MXFP4-only and stays focused on execution, KV, and tests. If maintainers prefer, #26869's author can take the NVFP4 portion and this PR the MXFP4 portion - a clean split along format lines
- mxfp6/mxfp8 dense quants are working in my tree as the next follow-up on the same `mxf8f6f4` plumbing
- Related KV cache: #6863 was the original 4-bit KV-cache feature request, #27362 (q3_K) is the precedent for adding a KV-cache type, #28633 proposes defaulting FA_ALL_QUANTS on for 4-bit KV, and #27109 reports a 4-bit-KV prefill collapse on a qwen35 hybrid (RTX 3090) - the mxfp4 KV data above shows no such collapse
- Related: #26989 requests the sibling NVFP4 KV cache, #19662 is the sm_120 block-scale mma build issue, and #26704 is adjacent SM120 MoE prefill work for MXFP4/NVFP4

## Requirements

- I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES - I used [pi.dev](https://pi.dev) and qwen3.6-27b and then qwen3.8-27b hosted using llama.cpp, running the code in this branch
