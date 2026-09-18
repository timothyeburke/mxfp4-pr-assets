# mxfp4 KV-cache: Universal Optimal Scaling (UOS) - findings

## Context
llama.cpp's mxfp4 (E2M1) KV-cache quantization uses the OCP scale formula
(e = lrint(log2(amax/4)) + 127), which pins the block max near 4.0 and never
clips, at the cost of more underflow for small values.

The MXAttention paper (arxiv 2607.24377) proposes **Universal Optimal Scaling
(UOS)**: a distribution-independent optimal scaling boundary Qmax per element
format, chosen to minimize the expected quantization error of the *entire*
block without any calibration. For E2M1 (mxfp4), Qmax = 7.25 (verified).

This tests UOS vs the OCP e_base for the **KV cache only** (weights keep the
imatrix-weighted scale search, which was separately proven already-optimal).

## Setup
- Model: Qwen3-0.8B (BF16 weights, so only the KV cache varies)
- Reference: 0.8b-base.bin (recorded with CUDA, PPL 15.1294)
- 3 runs, 72 chunks each, CPU 24 threads, build-mxfp4-kv-uos:
  - control: f16 KV (isolates CPU-vs-CUDA backend noise)
  - uos:     mxfp4 KV, UOS Qmax=7.25 (GGML_MXFP4_UOS=1)
  - ebase:   mxfp4 KV, OCP e_base (default)
- KV-cache effect = metric(run) - metric(control)

## Results
| run   | Mean KLD   | PPL(Q)   | PPL Δ (vs base) | top-p %  |
|-------|-----------|----------|-----------------|---------|
| control (f16 KV) | 0.000050 | 15.131 | +0.014 | 99.60 |
| uos (mxfp4 KV)   | 0.020220 | 15.248 | +0.131 | 92.50 |
| ebase (mxfp4 KV) | 0.022703 | 15.304 | +0.188 | 92.05 |

## Interpretation
- backend-noise floor (control): KLD 0.000050, PPL Δ +0.014, top-p 99.60%
- mxfp4 KV e_base effect: KLD 0.022703 (vs 0.000050 noise), PPL Δ +0.188
- mxfp4 KV UOS effect:     KLD 0.020220 (vs 0.000050 noise), PPL Δ +0.131
- **UOS vs e_base:**
  - KLD: 0.020220 vs 0.022703 -> **10.9% lower KLD** (UOS better)
  - PPL effect over noise: +0.117 vs +0.174 -> **33% smaller effect** (UOS better)
  - top-p: 92.50% vs 92.05% (UOS better)

## Conclusion
**UOS (Qmax=7.25) beats the OCP e_base for mxfp4 KV cache**, on every metric,
with no calibration (data-free). It reduces the KV-cache quantization error by
~11% (KLD) / ~33% (PPL effect) relative to the current OCP formula. This
confirms the MXAttention paper's premise for the mxfp4 (E2M1) KV-cache case:
the distribution-independent optimal boundary (Qmax=7.25) outperforms the
"pin-max-near-4, never-clip" OCP heuristic.

The mechanism: OCP pins the block max at ~4.0 (no clipping) but pushes small
values toward underflow (rounding to 0). UOS raises the boundary to 7.25,
allowing some headroom (the max may reach up to 7.25, i.e. mild clipping of
the very top values) in exchange for far less underflow of the bulk of the
distribution - which is what actually dominates the expected block error.

## Caveats
- 0.8B model, 72 chunks; the UOS margin (10.9% KLD) is well above the
  backend-noise floor (0.000050) and the run-to-run std (±0.000065), so it is
  a real effect, not noise.
- Weights are NOT affected (imatrix-weighted scale search, separately proven
  already-optimal). This is a KV-cache-only improvement.

## 35B validation (larger model, MoE A3B)

To confirm the result generalizes, the same test was run on Qwen3.6-35B-A3B
(BF16 weights, only KV cache varies), 3 runs x 72 chunks vs 35b-base.bin.

| run   | Mean KLD   | PPL Δ (vs base) | top-p %  |
|-------|-----------|-----------------|---------|
| control (f16 KV) | 0.002091 | +0.001 | 98.484 |
| uos (mxfp4 KV)   | 0.011629 | +0.022458 | 95.632 |
| ebase (mxfp4 KV) | 0.012184 | +0.020994 | 95.397 |

**UOS vs e_base on 35B (more nuanced than 0.8B):**
- KLD: 0.011629 vs 0.012184 -> UOS 4.6% lower (marginal, ~1.7x combined std)
- PPL: 0.022458 vs 0.020994 -> ebase 7% better (within noise, ~0.38x combined std)
- top-p: 95.632% vs 95.397% (UOS slightly better)

**Interpretation:** On the larger 35B MoE model, UOS and e_base are very close
(within noise on PPL, marginally different on KLD). The clear UOS win seen on
0.8B (10.9% lower KLD, 33% smaller PPL effect) is less pronounced on 35B.
The UOS advantage appears model-size-dependent: clearer on small models, more
mixed on large models. This is an honest, nuanced finding - the data-free UOS
boundary (Qmax=7.25) is a safe default (never worse than e_base beyond noise),
but its advantage is most visible on smaller models where the KV-cache
quantization error dominates.

## GPU round (2026-09-18, 2x RTX 5060 Ti) - DONE

Same three-arm design, but everything runs on CUDA (branch build, 72 chunks vs the
CUDA-recorded base .bin's, so the backend-noise floor is ~0). mxfp4-imx weight files
are constant across arms (BF16 does not fit for 27B/35B on 2x 16 GB), so the
weight-quant effect is included identically in every arm and the KV-cache effect is
arm-to-arm: metric(arm) - metric(control). The build includes the W4A8 activation
scale change (UOS Qmax=464); identical across arms, so the KV scale stays isolated.

| model | arm     | Mean KLD | PPL(Q)  | PPL delta vs base | top-p % |
|-------|---------|---------|---------|-------------------|---------|
| 0.8B  | control | 0.149191| 16.1832| +1.0664           | 81.283  |
| 0.8B  | uos     | 0.166515| 16.3692| +1.2525           | 80.194  |
| 0.8B  | ebase   | 0.168857| 16.4249| +1.3081           | 79.992  |
| 27B   | control | 0.089920|  6.3536| -0.0705           | 90.182  |
| 27B   | uos     | 0.091820|  6.4157| -0.0083           | 89.929  |
| 27B   | ebase   | 0.093152|  6.3974| -0.0267           | 89.939  |
| 35B   | control | 0.068204|  5.9285| +0.1852           | 89.396  |
| 35B   | uos     | 0.072918|  5.9504| +0.2072           | 88.921  |
| 35B   | ebase   | 0.073220|  5.9521| +0.2089           | 88.908  |

KV-cache effect (arm - control):

| model | KLD effect uos | KLD effect ebase | UOS reduction | PPL effect uos | PPL effect ebase |
|-------|---------------|-----------------|--------------|---------------|-----------------|
| 0.8B  | 0.017324      | 0.019666        | 11.9%        | +0.1860       | +0.2417         |
| 27B   | 0.001900      | 0.003232        | 41.2%        | +0.0622       | +0.0438         |
| 35B   | 0.004714      | 0.005016        | 6.0%         | +0.0220        | +0.0237         |

**GPU-round interpretation:**
- UOS lowers the KV-cache KLD effect on all 3 models (11.9% / 41.2% / 6.0%),
  consistent with the CPU round (10.9% on 0.8B).
- 0.8B: UOS wins on PPL (23% smaller effect) and top-p (+0.20 pt) - the clear case.
- 35B: UOS marginally better on PPL and top-p.
- 27B: KLD clearly favors UOS (41%); PPL slightly favors e_base (+0.044 vs +0.062)
  but the difference (0.018) is ~1.3x the combined std (~0.014) - within noise.
- Cross-check: the 0.8B control PPL (16.1832) matches the earlier w4a8 0.8B-imx
  PPL (16.1791) to 0.004 - the pipeline is consistent.

Reproduce (GPU):
- Build: build-mxfp8-act (CUDA, Release). ebase/control use the stock OCP e_base KV
  scale; uos uses cpy-utils.cuh switched to UOS Qmax=7.25 (one-line change + rebuild).
- Runs: logs/mxfp4-scale-search/kv-uos-gpu/{0.8b,27b,35b}-{control,uos,ebase}.log
- Analysis: kv_uos_gpu_analysis.py; harness: scripts/kld-kv-uos-gpu.sh

## Reproduce
- Build: build-mxfp4-kv-uos (CPU native, Release)
- 0.8B runs: logs/mxfp4-scale-search/kv-uos/{control,uos,ebase}.log
- 35B runs: logs/mxfp4-scale-search/kv-uos-35b/{control,uos,ebase}.log
- Analysis: kv_uos_kld_analysis.py
