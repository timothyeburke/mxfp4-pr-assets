---
license: apache-2.0
library_name: gguf
pipeline_tag: image-text-to-text
base_model: ggml-org/Qwen3.6-35B-A3B-GGUF
base_model_relation: quantized
tags:
- gguf
- mxfp4
- imatrix
- quantized
---

# Qwen3.6-35B-A3B - MXFP4 (imatrix)

4.25-bit MXFP4 quantization of [ggml-org/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF)
(`Qwen3.6-35B-A3B-BF16.gguf`), produced with imatrix weighting. MXFP4 means e2m1 values with one ue8m0 (e8m0) scale per
32-value block, as in the OCP microscaling spec. 733 tensors, `qwen35moe` architecture.

## Provenance

| step | artifact | produced by |
|---|---|---|
| 1 | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) @ `995ad96eacd98c81ed38be0c5b274b04031597b0` | original release |
| 2 | [ggml-org/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF) -> `Qwen3.6-35B-A3B-BF16.gguf` | ggml-org: `convert_hf_to_gguf.py --outtype bf16 --no-mtp --model-name Qwen3.6-35B-A3B` (see `convert.log` and `.src_sha` in that repo) |
| 3 | `35b.imatrix` (this repo) | `llama-imatrix -m Qwen3.6-35B-A3B-BF16.gguf -f wikitext-2-raw/wiki.train.raw -o 35b.imatrix -ngl 999` |
| 4 | `35b-mxfp4-imx.gguf` (this repo) | `llama-quantize --imatrix 35b.imatrix Qwen3.6-35B-A3B-BF16.gguf 35b-mxfp4-imx.gguf MXFP4` |

Steps 1-2 are ggml-org's, steps 3-4 are ours, and nothing besides the files above is an input. Step 3 ran
over 4894 chunks of `wiki.train.raw` at `n_ctx=512`, `batch_size=2048`, `n_seq=4`.

Both files in this repo were built by llama.cpp `0.5.0-dev` (build 3666, commit `0c666e561`) from the fork
in [timlikesai/llama.cpp#14](https://github.com/timlikesai/llama.cpp/pull/14), which adds the dense MXFP4
file type, the imatrix-weighted scale search and the MXFP4 KV cache path. That revision is what writes
`general.file_type = 42` (`LLAMA_FTYPE_MOSTLY_MXFP4` there); master has since renamed the ftype, so a newer
`llama-gguf` prints a different number for the same file.

Metadata recorded inside `35b-mxfp4-imx.gguf`:

- `general.architecture = qwen35moe`, `general.name = Qwen3.6-35B-A3B`, `general.file_type = 42`, `general.quantization_version = 2`; the base was converted with `--no-mtp`, so no `nextn.*` tensors are present
- Every `*_exps.weight` expert tensor is MXFP4 too (430 of the 733 tensors), so the MoE FFNs take the same path as dense ones
- `quantize.imatrix.dataset = wikitext-2-raw/wiki.train.raw`, `quantize.imatrix.chunks_count = 4894`, `quantize.imatrix.entries_count = 510`
- `quantize.imatrix.file` - the build-machine path of the imatrix file that is shipped in this repo

## Files

- `35b-mxfp4-imx.gguf` - MXFP4 weights (4.25 bpw, block-scale e8m0), OCP scale search with imatrix weighting
- `35b.imatrix` - the imatrix file from `wiki.train.raw` used by that scale search, so step 4 is reproducible
  without re-running step 3

Scale settings: weights use the OCP scale search (target 4.0); the MXFP4 KV cache path uses UOS scaling
(e = ceil(log2(amax/7.25))+127, MXAttention) as default.

## Checksums

| file | bytes | sha256 |
|---|---:|---|
| `35b-mxfp4-imx.gguf` | 19,041,835,296 | `3958f170a163e2fe238611ce396bcaac0a03f7ec3faa7486dda057ce0cd4c909` |
| `35b.imatrix` | 192,223,904 | `fc7d441e808fb837ccdab7760ce4dbeed4cc0edfb6eca4da81ff98e6b425beba` |

## Run it

```bash
llama-server -hf timlikesai/Qwen3.6-35B-A3B-MXFP4
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
| 4-bit (W4A4) | 0.1692 | 6.4741 | 82.615 | 5772 | 247.1 |
| FP8 (W4A8, default) | 0.0681 | 5.9312 | 89.316 | 5700 | 247.3 |
| q8_1 | 0.0656 | 5.9189 | 89.543 | 5249 | 247.1 |

Full three-model comparison and the exact repro commands:
https://github.com/timlikesai/llama.cpp/pull/15

Related: [#14](https://github.com/timlikesai/llama.cpp/pull/14) (MXFP4 file type, imatrix scale search,
KV cache) and [#15](https://github.com/timlikesai/llama.cpp/pull/15) (CUDA W4A8 for these weights).
