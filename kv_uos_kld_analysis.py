#!/usr/bin/env python3
"""
Parse the 3 KV-cache UOS KLD logs (control / uos / ebase) and compute the
isolated KV-cache quantization effect.

  control: BF16 model + f16 KV  (CPU build) vs base (CUDA)  -> backend noise floor
  uos:     BF16 model + mxfp4 KV (UOS Qmax=7.25)            -> UOS effect
  ebase:   BF16 model + mxfp4 KV (OCP e_base)               -> current effect

KV-cache effect (vs no-quant KV) = KLD(run) - KLD(control)
UOS improvement over e_base       = KLD(ebase) - KLD(uos)   (positive = UOS better)
"""
import re, sys, os

L = "/home/tim/code/services/repos/logs/mxfp4-scale-search/kv-uos"

def parse(name):
    p = os.path.join(L, f"{name}.log")
    if not os.path.exists(p):
        return None
    txt = open(p).read()
    # find the "Mean KLD" / "Mean PPL" summary lines (the final aggregate)
    kld = ppl = topp = None
    for m in re.finditer(r"Mean\s+KLD\s*:\s*([0-9.]+)", txt):
        kld = float(m.group(1))
    for m in re.finditer(r"Mean PPL\(Q\)\s*:\s*([0-9.]+)", txt):
        ppl = float(m.group(1))
    for m in re.finditer(r"Same top p\s*:\s*([0-9.]+)", txt):
        topp = float(m.group(1))
    # also grab the per-chunk KLD mean (the "Mean KLD" may be labeled differently)
    if kld is None:
        for m in re.finditer(r"KLD\s+([0-9]+\.[0-9]+)\s*±", txt):
            pass  # per-chunk; skip
    return {"kld": kld, "ppl": ppl, "topp": topp}

def main():
    names = ["control", "uos", "ebase"]
    res = {n: parse(n) for n in names}
    print(f"{'run':10s} {'KLD':>12s} {'PPL':>12s} {'top-p%':>10s}")
    for n in names:
        r = res[n]
        if r is None:
            print(f"{n:10s} (not finished yet)")
            continue
        kld = f"{r['kld']:.6f}" if r['kld'] is not None else "  (running)"
        ppl = f"{r['ppl']:.4f}" if r['ppl'] is not None else "  (running)"
        tp  = f"{r['topp']:.3f}" if r['topp'] is not None else "  (running)"
        print(f"{n:10s} {kld:>12s} {ppl:>12s} {tp:>10s}")
    print()
    c, u, e = res["control"], res["uos"], res["ebase"]
    if c and c["kld"] is not None and u and u["kld"] is not None and e and e["kld"] is not None:
        print(f"backend-noise floor (control KLD): {c['kld']:.6f}")
        print(f"mxfp4 KV e_base  KLD: {e['kld']:.6f}  -> effect over control: {e['kld']-c['kld']:+.6f}")
        print(f"mxfp4 KV UOS     KLD: {u['kld']:.6f}  -> effect over control: {u['kld']-c['kld']:+.6f}")
        print(f"UOS vs e_base:    {u['kld']:.6f} vs {e['kld']:.6f}  (UOS-ebase = {u['kld']-e['kld']:+.6f}; negative = UOS better)")
    else:
        print("(waiting for all 3 runs to finish)")

if __name__ == "__main__":
    main()
