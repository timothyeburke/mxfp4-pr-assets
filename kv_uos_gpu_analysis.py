#!/usr/bin/env python3
"""
Parse the GPU KV-cache UOS KLD logs (control / uos / ebase, 3 models) from
logs/mxfp4-scale-search/kv-uos-gpu/ and compute the isolated KV-cache effect.

  control: mxfp4 weights + f16 KV  (same-build reference; GPU noise floor ~0)
  uos:     mxfp4 weights + mxfp4 KV, UOS Qmax=7.25
  ebase:   mxfp4 weights + mxfp4 KV, OCP e_base

All KLD/PPL are vs the CUDA-recorded base .bin, so the weight-quant effect is
included identically in every arm; the KV-cache effect is arm-to-arm:
  KV effect  = metric(arm) - metric(control)
  UOS gain   = KLD(ebase) - KLD(uos)   (positive = UOS better)
"""
import os, re, sys

L = "/home/tim/code/services/repos/logs/mxfp4-scale-search/kv-uos-gpu"
MODELS = ["0.8b", "27b", "35b"]
ARMS = ["control", "uos", "ebase"]

def parse(model, arm):
    p = os.path.join(L, f"{model}-{arm}.log")
    if not os.path.exists(p):
        return None
    txt = open(p).read()
    def last(pattern):
        m = None
        for m in re.finditer(pattern, txt):
            pass
        return float(m.group(1)) if m else None
    return {
        "kld":  last(r"Mean\s+KLD\s*:\s*([0-9.]+)"),
        "ppl":  last(r"Mean PPL\(Q\)\s*:\s*([0-9.]+)"),
        "pplb": last(r"Mean PPL\(base\)\s*:\s*([0-9.]+)"),
        "topp": last(r"Same top p\s*:\s*([0-9.]+)"),
    }

def main():
    print(f"{'model':6s} {'arm':8s} {'KLD':>10s} {'PPL(Q)':>9s} {'PPL d':>8s} {'top-p%':>8s}")
    res = {}
    for m in MODELS:
        for a in ARMS:
            r = parse(m, a)
            res[(m, a)] = r
            if r is None:
                print(f"{m:6s} {a:8s}  (missing)")
                continue
            d = r["ppl"] - r["pplb"] if r["ppl"] and r["pplb"] else None
            ds = f"{d:+.4f}" if d is not None else "-"
            print(f"{m:6s} {a:8s} {r['kld']:10.6f} {r['ppl']:9.4f} {ds:>8s} {r['topp']:8.3f}")
    print()
    print("KV-cache effect (arm - control) and UOS gain over e_base:")
    for m in MODELS:
        c, u, e = res.get((m, "control")), res.get((m, "uos")), res.get((m, "ebase"))
        if not (c and u and e):
            print(f"{m:6s}  (incomplete)")
            continue
        print(f"{m:6s}  uos:  dKLD={u['kld']-c['kld']:+.6f}  dPPL={u['ppl']-c['ppl']:+.4f}  "
              f"ebase: dKLD={e['kld']-c['kld']:+.6f}  dPPL={e['ppl']-c['ppl']:+.4f}  "
              f"UOS gain KLD={e['kld']-u['kld']:+.6f}")

if __name__ == "__main__":
    main()
