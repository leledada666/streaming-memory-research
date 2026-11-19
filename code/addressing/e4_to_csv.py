# -*- coding: utf-8 -*-
"""Flatten an e4_addressing JSON report into CSV (one row per arm/cond/seed).

Usage: python e4_to_csv.py <report.json> [out.csv]
Zero-shot control evals are emitted as extra rows with kind='zero_shot'.
"""
import csv
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python e4_to_csv.py <report.json> [out.csv]")
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".csv")
    rep = json.loads(src.read_text(encoding="utf-8"))

    buckets = []
    for r in rep.get("runs", []):
        for k in r["eval"]["buckets"]:
            if k not in buckets:
                buckets.append(k)
    buckets.sort(key=lambda s: int(s.split("-")[0]))

    cols = (["kind", "arm", "cond", "seed", "params"]
            + [f"acc_{b}" for b in buckets]
            + ["acc_overall"]
            + [f"attn_mass_{b}" for b in buckets])

    rows = []
    for r in rep.get("runs", []):
        e = r["eval"]
        row = {"kind": "trained", "arm": r["arm"], "cond": r["cond"],
               "seed": r["seed"], "params": r.get("params")}
        for b in buckets:
            row[f"acc_{b}"] = e["buckets"].get(b, "")
        row["acc_overall"] = e.get("overall", "")
        am = e.get("attn_mass_at_s") or {}
        for b in buckets:
            row[f"attn_mass_{b}"] = am.get(b, "")
        rows.append(row)
        for c2, z in (r.get("zero_shot") or {}).items():
            row = {"kind": "zero_shot", "arm": r["arm"], "cond": c2,
                   "seed": r["seed"], "params": r.get("params")}
            for b in buckets:
                row[f"acc_{b}"] = z["buckets"].get(b, "")
            row["acc_overall"] = z.get("overall", "")
            rows.append(row)

    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {dst} ({len(rows)} rows)")


if __name__ == "__main__":
    main()