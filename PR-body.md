## Overview

Adds a W4A8 path for MXFP4 `mul_mat` on Blackwell. `src1` is quantized to e4m3 with ue8m0 block
scales and multiplied against the e2m1 weights with the block-scaled `kind::mxf8f6f4` MMA
(`m16n8k32`). MXFP4 on Blackwell uses this path by default.

- `GGML_PREC_MXFP8` added to `ggml_prec`.
- `mma_block_scaled_fp4` takes a `prec_src1` template argument and emits
  `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0`.
  The existing `kind::mxf4` `m16n8k64` e2m1 x e2m1 instruction is kept for `GGML_PREC_Q4`.
- `ggml_cuda_mmq_get_prec_src1` returns `GGML_PREC_MXFP8` for MXFP4 on Blackwell unless Q4 or Q8
  is requested. NVFP4 is unchanged.
- `quantize_mmq_mxfp8` and its MoE scatter variant: one warp per 32-value block, lane `l` holds
  values `4l..4l+3`, `amax` reduced over the 8 lanes of a scale group, e8m0 scale
  `2^rint(log2(amax / 256))`, four e4m3 codes packed per lane.
- `ggml_cuda_mmq_load_tiles_mxfp4_mxfp8` loads the packed e2m1 weights and the ue8m0 scales into
  the SRAM tile; the dot kernel unpacks them into the `mxf8f6f4` A-fragment.
- MMQ Blackwell tile/SRAM config and template instances for the (MXFP4, MXFP8) pair. The `src1`
  stride is taken from `y_block_size / y_values_per_block`.
- `test-backend-ops`: W4A4 / W4A8 / Q8_1 cases for MXFP4. The nmse tolerance follows the
  effective precision, since `GGML_CUDA_MMQ_PREC` overrides the op parameter.

![W4A8 vs W4A4: KLD, PPL, same top-p, prefill pp4096, decode tg128](https://raw.githubusercontent.com/timothyeburke/mxfp4-pr-assets/master/w4a8-results.png)

<details>
<summary>Results</summary>

imatrix mxfp4 files. KLD and PPL from `llama-perplexity --kl-divergence --file wikitext-2
--context-size 4096` over 72 chunks against the BF16 base. Throughput from `llama-bench
--n-gpu-layers 999 --split-mode tensor --flash-attn on --n-prompt 4096 --n-predict 128
--n-repeat 5`. Accuracy is hardware-invariant; the 2x RTX 5070 Ti run agrees within rounding.

Accuracy:

| model | arm | KLD | PPL | top-p |
|---|---|---:|---:|---:|
| 0.8B  | w4a4 | 0.407429 | 21.3419 | 69.960 |
| 0.8B  | w4a8 | 0.149528 | 16.1806 | 81.248 |
| 0.8B  | q8   | 0.143151 | 16.0858 | 81.646 |
| 27B   | w4a4 | 0.183180 |  6.5662 | 84.104 |
| 27B   | w4a8 | 0.089938 |  6.3581 | 90.166 |
| 27B   | q8   | 0.085955 |  6.3446 | 90.500 |
| 35B   | w4a4 | 0.168713 |  6.4702 | 82.668 |
| 35B   | w4a8 | 0.068391 |  5.9333 | 89.356 |
| 35B   | q8   | 0.065711 |  5.9161 | 89.571 |

Throughput, 2x RTX 5070 Ti:

| model | arm | pp4096 | tg128 |
|---|---|---:|---:|
| 0.8B  | w4a4 | 22864 | 527.5 |
| 0.8B  | w4a8 | 22821 | 527.9 |
| 0.8B  | q8   | 22261 | 528.2 |
| 27B   | w4a4 |  2518 |  78.9 |
| 27B   | w4a8 |  2388 |  78.7 |
| 27B   | q8   |  2186 |  78.8 |
| 35B   | w4a4 |  5755 | 242.5 |
| 35B   | w4a8 |  5644 | 242.7 |
| 35B   | q8   |  5260 | 242.4 |

Throughput, 2x RTX 5060 Ti:

| model | arm | pp4096 | tg128 |
|---|---|---:|---:|
| 0.8B  | w4a4 | 21898 | 415.0 |
| 0.8B  | w4a8 | 21594 | 414.6 |
| 0.8B  | q8   | 20726 | 414.7 |
| 27B   | w4a4 |  2077 |  46.5 |
| 27B   | w4a8 |  1890 |  46.5 |
| 27B   | q8   |  1674 |  46.5 |
| 35B   | w4a4 |  4323 | 177.4 |
| 35B   | w4a8 |  4163 | 177.5 |
| 35B   | q8   |  3861 | 177.4 |

</details>

## Requirements

- I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES - [pi.dev](https://pi.dev) with qwen3.8-27b served by llama.cpp wrote
  the code, ran the measurements and generated the chart.
