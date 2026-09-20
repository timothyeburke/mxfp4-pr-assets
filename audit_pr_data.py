#!/usr/bin/env python3
"""Audit: every number in PR-body.md tables vs the data in charts.py.

Parses the markdown tables and compares against the chart data structures
(PPL, KL, KV, W4A5, KVDATA). Reports mismatches beyond display rounding.
Run: python3 audit_pr_data.py [PR-body.md]
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import charts  # noqa: E402

BODY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "PR-body.md")

MODEL_BY_QWEN = {"Qwen3.5-0.8B": "0.8B", "Qwen3.8-27B": "27B", "Qwen3.6-35B-A3B": "35B"}

errors = []


def close(a, b, tol, what):
    if a is None or b is None:
        if a is not b:
            errors.append(f"{what}: table {a} vs chart {b}")
        return
    if abs(a - b) > tol:
        errors.append(f"{what}: table {a} vs chart {b}")


def clean_name(s):
    s = s.strip().replace("*", "").strip()
    if "MoE recipe" in s or "moe" in s.lower():
        return "mxfp4_moe"
    return re.sub(r"\s*\(.+\)$", "", s).strip()


def table_rows(md, header_prefix):
    """Yield (model, cells) for markdown tables; model from the nearest [Qwen...] header above."""
    lines = md.splitlines()
    cur_model = None
    for i, ln in enumerate(lines):
        m = re.match(r"^\[?(Qwen[\w.-]+)\]?\(", ln)
        if m:
            cur_model = MODEL_BY_QWEN.get(m.group(1))
        if (ln.startswith("|") and header_prefix in ln and i + 1 < len(lines)
                and re.match(r"^\|[\s:|-]+\|$", lines[i + 1])):
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                yield cur_model, [c.strip().replace("*", "") for c in lines[j].strip("|").split("|")]
                j += 1


def fnum(s):
    if s in ("-", ""):
        return None
    return float(s)


def audit_ppl_tables(md):
    for model, cells in table_rows(md, "| quant | size (GB) | PPL |"):
        name = clean_name(cells[0])
        name_p = name.replace("mxfp4_moe", "mxfp4-moe")
        if model is None or not any(r[0] in (name, name_p) for r in charts.PPL.get(model, [])):
            errors.append(f"PPL {model}: no chart row for {name!r}")
            continue
        row = next(r for r in charts.PPL[model] if r[0] in (name, name_p))
        close(fnum(cells[1]), row[1], 0.05, f"PPL {model} {name} size")
        close(fnum(cells[2]), row[2], 0.0005, f"PPL {model} {name} imx")


def audit_kl_tables(md):
    for model, cells in table_rows(md, "| quant | KLD (imx) |"):
        name = clean_name(cells[0])
        if model is None or name not in charts.KL.get(model, {}):
            errors.append(f"KL {model}: no chart row for {name!r}")
            continue
        d = charts.KL[model][name]
        close(fnum(cells[1]), d[0], 0.000051, f"KL {model} {name} imx-kld")
        close(fnum(cells[2]), d[1], 0.0051, f"KL {model} {name} imx-topp")
        close(fnum(cells[3]), d[2], 0.000051, f"KL {model} {name} noimx-kld")
        close(fnum(cells[4]), d[3], 0.0051, f"KL {model} {name} noimx-topp")


def audit_kv_table(md):
    last = None
    for _model, cells in table_rows(md, "| model | KV | memory @100k |"):
        if cells[0]:
            last = MODEL_BY_QWEN.get(cells[0], cells[0])
        if last is None or last not in charts.KV:
            continue
        kv, mem = cells[1], fnum(cells[2].replace(" GiB", ""))
        d = charts.KV[last]
        close(mem, d["mem"][kv], 0.005, f"KV {last} {kv} mem")
        close(fnum(cells[4]), d["tg"][kv], 0.05, f"KV {last} {kv} gpu-tg")


def audit_w4a8_table(md):
    for _model, cells in table_rows(md, "| model | file | KLD W4A4 old |"):
        model = cells[0]
        vals = [fnum(c) for c in cells[2:7]]
        keys = ["w4a4_old", "w4a4_uos", "w4a8_256", "w4a8_343", "w4a8_464"]
        for v, k in zip(vals, keys):
            close(v, charts.W4A5[model][k][1], 0.0000005, f"W4A5 {model} {k} kld")


def audit_uos_table(md):
    for _model, cells in table_rows(md, "| model | control (f16 KV) |"):
        model = cells[0]
        d = charts.KVDATA[model]
        for i, part in enumerate(("control", "e_base", "uos")):
            kld, ppl, tp = (fnum(x) for x in cells[1 + i].split("/"))
            close(kld, d["kld"][i], 0.0000005, f"UOS {model} {part} kld")
            close(ppl, d["ppl"][i], 0.0005, f"UOS {model} {part} ppl")
            close(tp, d["topp"][i], 0.005, f"UOS {model} {part} topp")


def main():
    md = open(BODY).read()
    audit_ppl_tables(md)
    audit_kl_tables(md)
    audit_kv_table(md)
    audit_w4a8_table(md)
    audit_uos_table(md)
    if errors:
        print(f"FAIL: {len(errors)} mismatch(es):")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("PASS: all PR-body table values match charts.py data")


if __name__ == "__main__":
    main()
