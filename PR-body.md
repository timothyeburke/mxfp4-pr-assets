## Overview

Adds a W4A8 path for MXFP4 `mul_mat` on Blackwell. `src1` is quantized to e4m3 with ue8m0 block
scales and multiplied against the e2m1 weights with the block-scaled `kind::mxf8f6f4` MMA
(`m16n8k32`). MXFP4 on Blackwell uses this path by default.

This is the CUDA mixed-precision MMA piece of [timlikesai/llama.cpp#14](https://github.com/timlikesai/llama.cpp/pull/14),
which also carries the dense MXFP4 ftype, imatrix weight quantization and an MXFP4 KV cache. It is
split out so the change that raises MXFP4 accuracy can be reviewed on its own: it only affects how
`src1` is quantized and which MMA instruction runs, the weight format is unchanged, and it applies
to `master` without the rest of #14. Dense model support is planned for a subsequent PR.

- `GGML_PREC_MXFP8` added to `ggml_prec`. e4m3 is a distinct src1 rank from `GGML_PREC_Q8`:
  e4m3 codes with one ue8m0 power-of-two scale per 32 values, matching the MXFP4 weight scale
  groups, instead of q8_1 fp32 block scales. `ggml_prec` values are stored in `op_params` and
  read by the backend, so the new rank needs no change outside the enum, like `GGML_PREC_Q4`.
  `GGML_CUDA_MMQ_PREC` stays a debug override; the enum is how a graph asks for e4m3.
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

Models, quantized to MXFP4 with imatrix by [timlikesai/llama.cpp#14](https://github.com/timlikesai/llama.cpp/pull/14).
Each repo also carries the imatrix file that was used, so the quants are reproducible:

- [Qwen3.5-0.8B-MXFP4](https://huggingface.co/timlikesai/Qwen3.5-0.8B-MXFP4) (dense)
- [Qwen3.8-27B-MXFP4](https://huggingface.co/timlikesai/Qwen3.8-27B-MXFP4) (dense)
- [Qwen3.6-35B-A3B-MXFP4](https://huggingface.co/timlikesai/Qwen3.6-35B-A3B-MXFP4) (MoE, exercises `mul_mat_id`)

The tables are the `*-mxfp4-imx.gguf` file in each repo. Arms: `w4a4` is `master`, which is the current MXFP4 behavior;
`w4a8` is this branch at its default; `q8` is this branch with `GGML_CUDA_MMQ_PREC=q8`, which
selects q8_1 activations.

Accuracy, KLD and PPL against BF16 reference logits recorded from the unquantized base:

    llama-perplexity -m <model>-mxfp4-imx.gguf -f wikitext-2-raw/wiki.test.raw \
      --kl-divergence --kl-divergence-base <base>.bin \
      -c 4096 --chunks 72 -ngl 999 -sm tensor -fa on --cache-type-k f16 --cache-type-v f16

Throughput:

    llama-bench -m <model>-mxfp4-imx.gguf -p 4096 -n 128 -ngl 999 -sm tensor -fa on -r 5

Accuracy is hardware-invariant; the 2x RTX 5060 Ti run agrees within rounding.

Accuracy:

| model | arm | KLD | PPL | top-p |
|---|---|---:|---:|---:|
| 0.8B  | w4a4 | 0.4086 | 21.4186 | 70.207 |
| 0.8B  | w4a8 | 0.1494 | 16.1757 | 81.237 |
| 0.8B  | q8   | 0.1433 | 16.0827 | 81.640 |
| 27B   | w4a4 | 0.1859 |  6.5416 | 84.000 |
| 27B   | w4a8 | 0.0894 |  6.3558 | 90.217 |
| 27B   | q8   | 0.0862 |  6.3436 | 90.472 |
| 35B   | w4a4 | 0.1692 |  6.4566 | 82.561 |
| 35B   | w4a8 | 0.0682 |  5.9363 | 89.247 |
| 35B   | q8   | 0.0657 |  5.9126 | 89.538 |

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
