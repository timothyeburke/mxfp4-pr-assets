## Overview

Adds a W4A8 path for MXFP4 `mul_mat` on Blackwell. `src1` is quantized to e4m3 values with one ue8m0
scale per 32 values and multiplied against the e2m1 weights with the block-scaled MMA instruction
`kind::mxf8f6f4` (`m16n8k32`, `scale_vec::1X`). On Blackwell, MXFP4 layers use this path by default.
`GGML_CUDA_MMQ_PREC` keeps `q4` (e2m1 activations, the existing `kind::mxf4` instruction) and `q8`
(q8_1 activations), and adds `fp8` (this path).

This is the CUDA mixed-precision MMA piece of [timlikesai/llama.cpp#14](https://github.com/timlikesai/llama.cpp/pull/14),
which also carries the dense MXFP4 ftype, imatrix weight quantization and an MXFP4 KV cache. It is
split out so the change that raises MXFP4 accuracy can be reviewed on its own: it changes how `src1` is
quantized and which MMA instruction runs. The weight format is unchanged. It applies to `master` without
the rest of #14. Dense MXFP4 quantization is planned for a subsequent PR: master already runs dense
MXFP4 files, which is why two of the three models below are dense, but `LLAMA_FTYPE_MOSTLY_MXFP4_MOE`
is the only MXFP4 ftype that can produce them today.

- `ggml_prec` gains `GGML_PREC_FP8 = 35`. e4m3 is a distinct `src1` rank from `GGML_PREC_Q8`: e4m3
  codes with one ue8m0 power-of-two scale per 32 values, matching the MXFP4 weight scale groups,
  instead of q8_1 fp32 block scales. `ggml_prec` values are stored in `op_params` and read by the
  backend, so the new rank needs no change outside the enum, like `GGML_PREC_Q4`. `GGML_CUDA_MMQ_PREC`
  stays a debug override; the enum is how a graph asks for e4m3.
- `mma_block_scaled_fp4` takes a `prec_src1` template argument and emits
  `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0`
  for `GGML_PREC_FP8`. The `kind::mxf4` (`m16n8k64`, e2m1 x e2m1, ue8m0) and `kind::mxf4nvf4`
  (`m16n8k64`, e2m1 x e2m1, ue4m3) instructions already in that function keep their behavior.
- `ggml_cuda_mmq_get_prec_src1` returns `GGML_PREC_FP8` for MXFP4 on Blackwell unless Q4 or Q8 is
  requested. NVFP4 keeps its existing behavior, and off Blackwell FP4 layers keep q8_1 activations.
- `quantize_mmq_fp8` and its MoE scatter variant: one warp per 128-value block, lane `l` holds values
  `4l..4l+3`, `amax` reduced over the 8 lanes of a scale group, block scale
  `2^(rint(log2(amax)) - 8)` from `compute_e8m0_scale(amax, E4M3_EMAX)`, four e4m3 codes packed per lane.
- `ggml_cuda_mmq_load_tiles_mxfp4_fp8` writes the packed e2m1 weights and the ue8m0 scales into the
  SRAM tile; `ggml_cuda_mmq_vec_dot_mxfp4_fp8_mma` unpacks the weights into the `mxf8f6f4` A-fragment
  (e2m1 nibbles in bits `[2..5]` of each e4m3 slot) and issues the MMA.
- `block_fp8_mmq` with `QK_FP8_MMQ` and `GGML_CUDA_MMQ_SRAM_LAYOUT_FP8`: 128 e4m3 values plus one ue8m0
  scale per 32 values, `static_assert`-sized to `block_q8_1_mmq`. The FP8 config selects
  `K_vram = MMQ_ITER_K`, the same number of blocks per iteration as the q8_1 path.
- MMQ dispatch and template instances for the (MXFP4, `GGML_PREC_FP8`) pair, including `DECL_MMQ_CASE_FP8`
  in the generated instance files. The `src1` stride is taken from `y_block_size / y_values_per_block`.
- `test-backend-ops`: `MUL_MAT_W4A4`, `MUL_MAT_W4A8` and `MUL_MAT_W4A8_FP8` cases for `MUL_MAT` and
  `MUL_MAT_ID`. The nmse tolerance follows the effective precision, since `GGML_CUDA_MMQ_PREC`
  overrides the op parameter.
- `docs/build.md`: the `GGML_CUDA_MMQ_PREC` value list and the Blackwell default.

## Additional information

What the change does not touch: `nvfp4` weight handling and its ue4m3 scales; the MMQ config tables in
`mmq-config-blackwell.cuh`; `mul_mat_q_process_tile`, which is byte-identical to master; and the
existing `quantize_mmq_mxfp4` / `quantize_mmq_nvfp4` kernels together with `compute_e8m0_scale`'s e2m1
call site. `ggml/src/ggml-cuda/mmq.cuh` is a pure insertion. The diff is 529 lines across 12 files
(+476 / -53).

Correctness, 2x RTX 5070 Ti, `test-backend-ops -b CUDA0 -o "MUL_MAT.*"`:

| build | env | result |
|---|---|---|
| branch | `unset` | 4993/4993 tests passed |
| branch | `q4` | 4993/4993 tests passed |
| branch | `q8` | 4993/4993 tests passed |
| branch | `fp8` | 4993/4993 tests passed |
| branch | `zzz` | 4993/4993 tests passed |
| base | `unset` | 4989/4989 tests passed |

All cells below were measured on the 2x RTX 5070 Ti box.

![W4A8 vs W4A4: KLD, PPL, same top-p, prefill pp4096, decode tg128](https://raw.githubusercontent.com/timothyeburke/mxfp4-pr-assets/master/w4a8-results.png)

<details>
<summary>Results</summary>

Models, quantized to MXFP4 with imatrix by [timlikesai/llama.cpp#14](https://github.com/timlikesai/llama.cpp/pull/14).
Each repo also carries the imatrix file that was used, so the quants are reproducible:

- [Qwen3.5-0.8B-MXFP4](https://huggingface.co/timlikesai/Qwen3.5-0.8B-MXFP4) (dense)
- [Qwen3.8-27B-MXFP4](https://huggingface.co/timlikesai/Qwen3.8-27B-MXFP4) (dense)
- [Qwen3.6-35B-A3B-MXFP4](https://huggingface.co/timlikesai/Qwen3.6-35B-A3B-MXFP4) (MoE, exercises `mul_mat_id`)

The tables use the `*-mxfp4-imx.gguf` file of each repo. Arms: `mxfp4-mxfp4` is `master`, the current
MXFP4 behavior; `mxfp4-mxfp8` is this branch at its default; `mxfp4-q8` is this branch with
`GGML_CUDA_MMQ_PREC=q8`, which selects q8_1 activations. All three read the same weight files.

Accuracy, KLD and PPL against BF16 reference logits recorded from the unquantized base:

    llama-perplexity -m <model>-mxfp4-imx.gguf -f wikitext-2-raw/wiki.test.raw \
      --kl-divergence --kl-divergence-base <base>.bin \
      -c 4096 --chunks 72 -ngl 999 -sm tensor -fa on --cache-type-k f16 --cache-type-v f16

Throughput:

    llama-bench -m <model>-mxfp4-imx.gguf -p 4096 -n 128 -ngl 999 -sm tensor -fa on -r 5

Accuracy:

| model | arm | KLD | PPL | top-p |
|---|---|---:|---:|---:|
| 0.8B  | mxfp4-mxfp4 | 0.4079 | 21.3590 | 69.847 |
| 0.8B  | mxfp4-mxfp8 | 0.1493 | 16.1805 | 81.231 |
| 0.8B  | mxfp4-q8    | 0.1432 | 16.0846 | 81.639 |
| 27B   | mxfp4-mxfp4 | 0.1875 |  6.5477 | 83.958 |
| 27B   | mxfp4-mxfp8 | 0.0898 |  6.3524 | 90.153 |
| 27B   | mxfp4-q8    | 0.0860 |  6.3469 | 90.477 |
| 35B   | mxfp4-mxfp4 | 0.1693 |  6.4713 | 82.628 |
| 35B   | mxfp4-mxfp8 | 0.0683 |  5.9316 | 89.308 |
| 35B   | mxfp4-q8    | 0.0658 |  5.9175 | 89.548 |

Throughput results:

| model | arm | pp4096 | tg128 |
|---|---|---:|---:|
| 0.8B  | mxfp4-mxfp4 | 23935 | 540.8 |
| 0.8B  | mxfp4-mxfp8 | 23209 | 541.5 |
| 0.8B  | mxfp4-q8    | 23133 | 541.4 |
| 27B   | mxfp4-mxfp4 | 2560 | 79.0 |
| 27B   | mxfp4-mxfp8 | 2327 | 80.0 |
| 27B   | mxfp4-q8    | 2194 | 80.0 |
| 35B   | mxfp4-mxfp4 | 5892 | 248.7 |
| 35B   | mxfp4-mxfp8 | 5781 | 248.3 |
| 35B   | mxfp4-q8    | 5339 | 248.2 |

The 0.8B cells are noisy at `-r 5`. Re-run at `-r 15` on the same build: `mxfp4-mxfp4` pp4096 23872.82 ± 83.20, tg128 542.97 ± 3.41; `mxfp4-mxfp8` pp4096 23719.63 ± 83.31, tg128 542.55 ± 3.33; `mxfp4-q8` pp4096 23016.86 ± 107.72, tg128 542.64 ± 2.71.

</details>

## Requirements

- I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES - [pi.dev](https://pi.dev) with qwen3.8-27b served by llama.cpp wrote
  the code, ran the measurements and generated the chart.
