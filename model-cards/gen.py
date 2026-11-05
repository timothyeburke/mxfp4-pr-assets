import os

MODELS = [
 dict(repo="Qwen3.5-0.8B-MXFP4", name="Qwen3.5-0.8B", arch="qwen35", tens=335,
      gguf="0.8b-mxfp4-imx.gguf", gsize="558,371,904", gsha="087de261f16e8b6ee1b89df6e8534a917514e49225ba552e20cc4d19b26bc083",
      imx="0.8b.imatrix", isize="1,134,720", isha="783985d78f0b1fb2b6200ba0a438d5ada5f378120b87b56a32b90e98e6c0d3dc",
      entries=186, bf16="Qwen3.5-0.8B-BF16.gguf", sha_src="2fc06364715b967f1860aea9cf38778875588b17",
      exptype="", nomtp=False, extra="; it also records `general.base_model.0.name = Qwen3.5 0.8B Base` with `repo_url = https://huggingface.co/Qwen/Qwen3.5-0.8B-Base`, and keeps the `blk.24.nextn.*` MTP tensors that the BF16 base carries",
      acc=[("4-bit (W4A4)","0.4079","21.3590","69.847"),("FP8 (W4A8, default)","0.1493","16.1805","81.231"),("q8_1","0.1432","16.0846","81.639")],
      spd=[("4-bit (W4A4)","23053","537.0"),("FP8 (W4A8, default)","22690","535.9"),("q8_1","22154","536.6")]),
 dict(repo="Qwen3.8-27B-MXFP4", name="Qwen3.8-27B", arch="qwen35", tens=851,
      gguf="27b-mxfp4-imx.gguf", gsize="15,659,530,848", gsha="eaf5d19ae975cac1380cdc0fa2bb37e13c4b96abd2b10a12898b837b9e6b1248",
      imx="27b.imatrix", isize="13,642,656", isha="f7c94b667efc85fed5dba2041da09308d58065a527ff20e12a27160c4ad6513a",
      entries=496, bf16="Qwen3.8-27B-BF16.gguf", sha_src="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
      exptype="", nomtp=True, extra="; the base was converted with `--no-mtp`, so no `nextn.*` tensors are present",
      acc=[("4-bit (W4A4)","0.1875","6.5477","83.958"),("FP8 (W4A8, default)","0.0898","6.3525","90.157"),("q8_1","0.0860","6.3470","90.476")],
      spd=[("4-bit (W4A4)","2533","79.8"),("FP8 (W4A8, default)","2341","79.7"),("q8_1","2178","79.9")]),
 dict(repo="Qwen3.6-35B-A3B-MXFP4", name="Qwen3.6-35B-A3B", arch="qwen35moe", tens=733,
      gguf="35b-mxfp4-imx.gguf", gsize="19,041,835,296", gsha="3958f170a163e2fe238611ce396bcaac0a03f7ec3faa7486dda057ce0cd4c909",
      imx="35b.imatrix", isize="192,223,904", isha="fc7d441e808fb837ccdab7760ce4dbeed4cc0edfb6eca4da81ff98e6b425beba",
      entries=510, bf16="Qwen3.6-35B-A3B-BF16.gguf", sha_src="995ad96eacd98c81ed38be0c5b274b04031597b0",
      exptype="- Every `*_exps.weight` expert tensor is MXFP4 too (430 of the 733 tensors), so the MoE FFNs take the same path as dense ones\n",
      nomtp=True, extra="; the base was converted with `--no-mtp`, so no `nextn.*` tensors are present",
      acc=[("4-bit (W4A4)","0.1692","6.4741","82.615"),("FP8 (W4A8, default)","0.0681","5.9312","89.316"),("q8_1","0.0656","5.9189","89.543")],
      spd=[("4-bit (W4A4)","5772","247.1"),("FP8 (W4A8, default)","5700","247.3"),("q8_1","5249","247.1")]),
]

TPL = """---
license: apache-2.0
library_name: gguf
pipeline_tag: image-text-to-text
base_model: ggml-org/{name}-GGUF
base_model_relation: quantized
tags:
- gguf
- mxfp4
- imatrix
- quantized
---

# {name} - MXFP4 (imatrix)

4.25-bit MXFP4 quantization of [ggml-org/{name}-GGUF](https://huggingface.co/ggml-org/{name}-GGUF)
(`{bf16}`), produced with imatrix weighting. MXFP4 means e2m1 values with one ue8m0 (e8m0) scale per
32-value block, as in the OCP microscaling spec. {tens} tensors, `{arch}` architecture.

## Provenance

| step | artifact | produced by |
|---|---|---|
| 1 | [Qwen/{name}](https://huggingface.co/Qwen/{name}) @ `{sha_src}` | original release |
| 2 | [ggml-org/{name}-GGUF](https://huggingface.co/ggml-org/{name}-GGUF) -> `{bf16}` | ggml-org: `convert_hf_to_gguf.py --outtype bf16{nomtp} --model-name {name}` (see `convert.log` and `.src_sha` in that repo) |
| 3 | `{imx}` (this repo) | `llama-imatrix -m {bf16} -f wikitext-2-raw/wiki.train.raw -o {imx} -ngl 999` |
| 4 | `{gguf}` (this repo) | `llama-quantize --imatrix {imx} {bf16} {gguf} MXFP4` |

Steps 1-2 are ggml-org's, steps 3-4 are ours, and nothing besides the files above is an input. Step 3 ran
over 4894 chunks of `wiki.train.raw` at `n_ctx=512`, `batch_size=2048`, `n_seq=4`.

Both files in this repo were built by llama.cpp `0.5.0-dev` (build 3666, commit `0c666e561`) from the fork
in [timlikesai/llama.cpp#14](https://github.com/timlikesai/llama.cpp/pull/14), which adds the dense MXFP4
file type, the imatrix-weighted scale search and the MXFP4 KV cache path. That revision is what writes
`general.file_type = 42` (`LLAMA_FTYPE_MOSTLY_MXFP4` there); master has since renamed the ftype, so a newer
`llama-gguf` prints a different number for the same file.

Metadata recorded inside `{gguf}`:

- `general.architecture = {arch}`, `general.name = {name}`, `general.file_type = 42`, `general.quantization_version = 2`{extra}
{exptype}- `quantize.imatrix.dataset = wikitext-2-raw/wiki.train.raw`, `quantize.imatrix.chunks_count = 4894`, `quantize.imatrix.entries_count = {entries}`
- `quantize.imatrix.file` - the build-machine path of the imatrix file that is shipped in this repo

## Files

- `{gguf}` - MXFP4 weights (4.25 bpw, block-scale e8m0), OCP scale search with imatrix weighting
- `{imx}` - the imatrix file from `wiki.train.raw` used by that scale search, so step 4 is reproducible
  without re-running step 3

Scale settings: weights use the OCP scale search (target 4.0); the MXFP4 KV cache path uses UOS scaling
(e = ceil(log2(amax/7.25))+127, MXAttention) as default.

## Checksums

| file | bytes | sha256 |
|---|---:|---|
| `{gguf}` | {gsize} | `{gsha}` |
| `{imx}` | {isize} | `{isha}` |

## Run it

```bash
llama-server -hf timlikesai/{repo}
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
{accrows}

Full three-model comparison and the exact repro commands:
https://github.com/timlikesai/llama.cpp/pull/15

Related: [#14](https://github.com/timlikesai/llama.cpp/pull/14) (MXFP4 file type, imatrix scale search,
KV cache) and [#15](https://github.com/timlikesai/llama.cpp/pull/15) (CUDA W4A8 for these weights).
"""

for m in MODELS:
    nomtp = " --no-mtp" if m.pop("nomtp") else ""
    rows = [f"| {a} | {k} | {p} | {t} | {pp} | {tg} |"
            for (a, k, p, t), (_, pp, tg) in zip(m["acc"], m["spd"])]
    txt = TPL.format(nomtp=nomtp, accrows="\n".join(rows), **m)
    d = f"/tmp/hf-cards/{m['repo']}"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/README.md", "w").write(txt)
    print(f"wrote {d}/README.md ({len(txt)} bytes)")
