## Overview

Adds a W4A8 path for MXFP4 `mul_mat` on Blackwell. `src1` is quantized to e4m3 values with one ue8m0 scale per 32 values and multiplied against the e2m1 weights with the block-scaled MMA instruction `kind::mxf8f6f4` (`m16n8k32`, `scale_vec::1X`). On Blackwell, MXFP4 layers now use this path by default. `GGML_CUDA_MMQ_PREC` keeps `q4` (e2m1 activations, the existing `kind::mxf4` instruction) and `q8`
(q8_1 activations), and adds `fp8`.

- `GGML_PREC_FP8 = 35`. e4m3 `src1` with one ue8m0 power-of-two scale per 32 values
- `mma_block_scaled` (renamed from `mma_block_scaled_fp4`), takes a `prec_src1` template argument and emits `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0` for `GGML_PREC_FP8`
- `ggml_cuda_mmq_get_prec_src1` returns `GGML_PREC_FP8` for MXFP4 on Blackwell unless Q4 or Q8 is requested. NVFP4 keeps its existing behavior, and off Blackwell FP4 layers keep q8_1 activations.
- `quantize_mmq_fp8` and its MoE scatter variant: one warp per 128-value block reduced over the 8 lanes of a scale group
- `ggml_cuda_mmq_load_tiles_mxfp4_fp8` writes the packed e2m1 weights and the ue8m0 scales into the SRAM tile; `ggml_cuda_mmq_vec_dot_mxfp4_fp8_mma` unpacks the weights into the `mxf8f6f4` A-fragment and issues the MMA
- `block_fp8_mmq` with `QK_FP8_MMQ` and `GGML_CUDA_MMQ_SRAM_LAYOUT_FP8`: 128 e4m3 values plus one ue8m0 scale per 32 values
- `test-backend-ops`: `MUL_MAT_W4A8_FP8` and `MUL_MAT_ID_W4A8_FP8` cases next to the existing `MUL_MAT_W4A4` / `MUL_MAT_W4A8` ones - dense and MoE, MoE with and without broadcast `src1`, `nrows_x` with and without the MMQ fallback (`nrows_x % 128`), and a `k` that is not a multiple of a 128-value `src1` block
- `docs/build.md`: the `GGML_CUDA_MMQ_PREC` value list and the Blackwell default

## Additional information

Correctness, 2x RTX 5070 Ti, `test-backend-ops -b CUDA0 -o "MUL_MAT.*"` - the branch has 9 more cases than the base, the new FP4 activation ones:

| build | env | result |
|---|---|---|
| branch  | `unset` | 5051/5051 tests passed |
| branch  | `q4` | 5051/5051 tests passed |
| branch  | `q8` | 5051/5051 tests passed |
| branch  | `fp8` | 5051/5051 tests passed |
| branch  | `zzz` | 5051/5051 tests passed |
| master | `unset` | 5042/5042 tests passed |

All cells below were measured on the 2x RTX 5070 Ti box (driver 615.71.09), on one binary: the branch build `3ca37060e` on base `8bed7cfb9`. Only `GGML_CUDA_MMQ_PREC` changes between arms, and every arm reads the same weight file.

![W4A8 vs W4A4: KLD, PPL, same top-p, prefill pp4096, decode tg128, accuracy vs file size](https://raw.githubusercontent.com/timothyeburke/mxfp4-pr-assets/master/w4a8-results.png)

<details>
<summary>Results</summary>

Models, quantized to MXFP4 with imatrix by [timlikesai/llama.cpp#14](https://github.com/timlikesai/llama.cpp/pull/14).
Each repo also carries the imatrix file that was used, so the quants are reproducible:

- [Qwen3.5-0.8B-MXFP4](https://huggingface.co/timlikesai/Qwen3.5-0.8B-MXFP4) (dense)
- [Qwen3.8-27B-MXFP4](https://huggingface.co/timlikesai/Qwen3.8-27B-MXFP4) (dense)
- [Qwen3.6-35B-A3B-MXFP4](https://huggingface.co/timlikesai/Qwen3.6-35B-A3B-MXFP4) (MoE)

`mxfp4-mxfp4` is `GGML_CUDA_MMQ_PREC=q4` is master's current MXFP4 path, `mxfp4-mxfp8` is the branch default, and `mxfp4-q8` is `GGML_CUDA_MMQ_PREC=q8`.

Accuracy, KLD and PPL against BF16 reference logits recorded from the base:
```
    llama-perplexity -m <model>-mxfp4-imx.gguf -f wikitext-2-raw/wiki.test.raw \
      --kl-divergence --kl-divergence-base <base>.bin \
      -c 4096 --chunks 72 -ngl 999 -sm tensor -fa on --cache-type-k f16 --cache-type-v f16
```
Throughput:
```
    llama-bench -m <model>-mxfp4-imx.gguf -p 4096 -n 128 -ngl 999 -sm tensor -fa on -r 5
```

Accuracy:

| model | arm | KLD | PPL | same top-p |
|---|---|---:|---:|---:|
| 0.8B  | mxfp4-mxfp4 | 0.4079 | 21.3590 | 69.847 |
| 0.8B  | mxfp4-mxfp8 | 0.1493 | 16.1805 | 81.231 |
| 0.8B  | mxfp4-q8    | 0.1432 | 16.0846 | 81.639 |
| 27B   | mxfp4-mxfp4 | 0.1875 |  6.5477 | 83.958 |
| 27B   | mxfp4-mxfp8 | 0.0898 |  6.3524 | 90.153 |
| 27B   | mxfp4-q8    | 0.0860 |  6.3470 | 90.476 |
| 35B   | mxfp4-mxfp4 | 0.1693 |  6.4713 | 82.628 |
| 35B   | mxfp4-mxfp8 | 0.0683 |  5.9316 | 89.308 |
| 35B   | mxfp4-q8    | 0.0658 |  5.9175 | 89.548 |

`same top-p` is the `llama-perplexity` "Same top p" figure: the share of positions whose argmax equals the BF16 base model's argmax. It is not nucleus sampling.

Throughput results (t/s, mean ± sd over 5 reps):

| model | arm | pp4096 | tg128 |
|---|---|---:|---:|
| 0.8B  | mxfp4-mxfp4 | 23053.23 ± 85.55 | 537.00 ± 4.15 |
| 0.8B  | mxfp4-mxfp8 | 22689.54 ± 122.29 | 535.93 ± 4.97 |
| 0.8B  | mxfp4-q8    | 22153.55 ± 69.46 | 536.64 ± 4.79 |
| 27B   | mxfp4-mxfp4 | 2533.43 ± 1.45 | 79.75 ± 0.08 |
| 27B   | mxfp4-mxfp8 | 2340.93 ± 0.87 | 79.67 ± 0.11 |
| 27B   | mxfp4-q8    | 2178.46 ± 1.31 | 79.88 ± 0.06 |
| 35B   | mxfp4-mxfp4 | 5771.72 ± 10.46 | 247.12 ± 2.04 |
| 35B   | mxfp4-mxfp8 | 5699.58 ± 16.73 | 247.27 ± 2.06 |
| 35B   | mxfp4-q8    | 5249.46 ± 13.12 | 247.11 ± 2.00 |

</details>

## Requirements

- I have read and agree with the [contributing guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- AI usage disclosure: YES - [pi.dev](https://pi.dev) with qwen3.8-27b served by llama.cpp wrote the code and ran the first measurement rounds; the re-measurement, the chart and the code review pass for this revision ran on qwen3.8-flash-next. 
- I (the author) have reviewed every change and own them.

