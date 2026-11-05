---
license: apache-2.0
library_name: gguf
pipeline_tag: image-text-to-text
base_model: ggml-org/Qwen3.8-27B-GGUF
base_model_relation: quantized
tags:
- gguf
- mxfp4
- imatrix
- quantized
---

# Qwen3.8-27B - MXFP4 (imatrix)

4.25-bit MXFP4 quantization of [ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF)
(`Qwen3.8-27B-BF16.gguf`), produced with imatrix weighting. MXFP4 means e2m1 values with one ue8m0 (e8m0) scale per
32-value block, as in the OCP microscaling spec. 851 tensors, `qwen35` architecture.

## Provenance

| step | artifact | produced by |
|---|---|---|
| 1 | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | original release |
| 2 | [ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) -> `Qwen3.8-27B-BF16.gguf` | ggml-org: `convert_hf_to_gguf.py --outtype bf16 --no-mtp --model-name Qwen3.8-27B` (see `convert.log` and `.src_sha` in that repo) |
| 3 | `27b.imatrix` (this repo) | `llama-imatrix -m Qwen3.8-27B-BF16.gguf -f wikitext-2-raw/wiki.train.raw -o 27b.imatrix -ngl 999` |
| 4 | `27b-mxfp4-imx.gguf` (this repo) | `llama-quantize --imatrix 27b.imatrix Qwen3.8-27B-BF16.gguf 27b-mxfp4-imx.gguf MXFP4` |

Steps 1-2 are ggml-org's, steps 3-4 are ours, and nothing besides the files above is an input. Step 3 ran
over 4894 chunks of `wiki.train.raw` at `n_ctx=512`, `batch_size=2048`, `n_seq=4`.

Both files in this repo were built by llama.cpp `0.5.0-dev` (build 3666, commit `0c666e561`) from the fork
in [timlikesai/llama.cpp#14](https://github.com/timlikesai/llama.cpp/pull/14), which adds the dense MXFP4
file type, the imatrix-weighted scale search and the MXFP4 KV cache path. That revision is what writes
`general.file_type = 42` (`LLAMA_FTYPE_MOSTLY_MXFP4` there); master has since renamed the ftype, so a newer
`llama-gguf` prints a different number for the same file.

Metadata recorded inside `27b-mxfp4-imx.gguf`:

- `general.architecture = qwen35`, `general.name = Qwen3.8-27B`, `general.file_type = 42`, `general.quantization_version = 2`; the base was converted with `--no-mtp`, so no `nextn.*` tensors are present
- `quantize.imatrix.dataset = wikitext-2-raw/wiki.train.raw`, `quantize.imatrix.chunks_count = 4894`, `quantize.imatrix.entries_count = 496`
- `quantize.imatrix.file` - the build-machine path of the imatrix file that is shipped in this repo

## Files

- `27b-mxfp4-imx.gguf` - MXFP4 weights (4.25 bpw, block-scale e8m0), OCP scale search with imatrix weighting
- `27b.imatrix` - the imatrix file from `wiki.train.raw` used by that scale search, so step 4 is reproducible
  without re-running step 3

Scale settings: weights use the OCP scale search (target 4.0); the MXFP4 KV cache path uses UOS scaling
(e = ceil(log2(amax/7.25))+127, MXAttention) as default.

## Checksums

| file | bytes | sha256 |
|---|---:|---|
| `27b-mxfp4-imx.gguf` | 15,659,530,848 | `eaf5d19ae975cac1380cdc0fa2bb37e13c4b96abd2b10a12898b837b9e6b1248` |
| `27b.imatrix` | 13,642,656 | `f7c94b667efc85fed5dba2041da09308d58065a527ff20e12a27160c4ad6513a` |

## Run it

```bash
llama-server -hf timlikesai/Qwen3.8-27B-MXFP4
```

On consumer Blackwell (sm_120a, RTX 50 series) with [timlikesai/llama.cpp#15](https://github.com/timlikesai/llama.cpp/pull/15)
these layers default to W4A8: FP8 e4m3 activations with ue8m0 block scales through the `kind::mxf8f6f4`
block-scaled MMA, which recovers most of the accuracy of q8_1 activations at a fraction of its prefill
cost. `GGML_CUDA_MMQ_PREC=q4` selects 4-bit activations, `q8` selects q8_1.

Measured on 2x RTX 5070 Ti with the file in this repo: wikitext-2 test, `n_ctx=4096`, 72 chunks, KLD and
PPL against the BF16 base of step 2 (`llama-perplexity`), throughput from `llama-bench -p 4096 -n 128 -r 5`
(t/s depends only on tensor shapes, so it is the same for any re-quantization of this base). The 4-bit row
is what current master does with these weights; the FP8 and q8_1 rows need the #15 build, which is where
`GGML_CUDA_MMQ_PREC` comes from:

| activation tier | KLD | PPL | same top-p | pp4096 t/s | tg128 t/s |
|---|---:|---:|---:|---:|---:|
| 4-bit (W4A4) | 0.1875 | 6.5477 | 83.958 | 2533 | 79.8 |
| FP8 (W4A8, default) | 0.0898 | 6.3525 | 90.157 | 2341 | 79.7 |
| q8_1 | 0.0860 | 6.3470 | 90.476 | 2178 | 79.9 |

Full three-model comparison and the exact repro commands:
https://github.com/timlikesai/llama.cpp/pull/15

Related: [#14](https://github.com/timlikesai/llama.cpp/pull/14) (MXFP4 file type, imatrix scale search,
KV cache) and [#15](https://github.com/timlikesai/llama.cpp/pull/15) (CUDA W4A8 for these weights).
