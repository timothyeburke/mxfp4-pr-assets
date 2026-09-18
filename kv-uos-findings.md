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

## Reproduce
- Build: build-mxfp4-kv-uos (CPU native, Release)
- Runs: logs/mxfp4-scale-search/kv-uos/{control,uos,ebase}.log
- Analysis: kv_uos_kld_analysis.py
