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
--n-repeat 5`.

Accuracy:

| model | KLD W4A4 | KLD W4A8 | PPL W4A4 | PPL W4A8 | top-p W4A4 | top-p W4A8 |
|---|---:|---:|---:|---:|---:|---:|
| 0.8B  | 0.407429 | 0.149528 | 21.3419 | 16.1806 | 69.960 | 81.248 |
| 27B   | 0.183180 | 0.089938 |  6.5662 |  6.3581 | 84.104 | 90.166 |
| 35B   | 0.168713 | 0.068391 |  6.4702 |  5.9333 | 82.668 | 89.356 |

Throughput, 2x RTX 5060 Ti:

| model | pp4096 W4A4 | pp4096 W4A8 | tg128 W4A4 | tg128 W4A8 |
|---|---:|---:|---:|---:|
| 0.8B  | 21951 | 21645 | 410.6 | 410.3 |
| 27B   |  2093 |  1890 |  46.5 |  46.5 |
| 35B   |  4431 |  4169 | 176.6 | 176.4 |

Throughput, 2x RTX 5070 Ti:

| model | pp4096 W4A4 | pp4096 W4A8 | tg128 W4A4 | tg128 W4A8 |
|---|---:|---:|---:|---:|
| 0.8B  | 23109 | 22933 | 529.0 | 527.9 |
| 27B   |  2520 |  2389 |  78.9 |  78.7 |
| 35B   |  5793 |  5666 | 242.8 | 242.6 |

</details>

## Requirements

- I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES - [pi.dev](https://pi.dev) with qwen3.8-27b served by llama.cpp wrote
  the code, ran the measurements and generated the chart.
