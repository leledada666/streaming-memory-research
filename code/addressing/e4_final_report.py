# -*- coding: utf-8 -*-
"""Merge E4 reports, write JSON/CSV/Markdown, and apply the pre-registered verdict."""
import csv, glob, json, statistics as st
from collections import defaultdict
from pathlib import Path

OUT = Path(r"C:\Users\lele\AppData\Local\Temp\opencode\e4_full")
BUCKETS = ["1-8", "9-16", "17-32", "33-64"]

def load_runs():
    rows = {}
    for p in sorted(OUT.glob("e4_*.json")):
        try:
            rep = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in rep.get("runs", []):
            key = ("trained", r["arm"], r["cond"], r["seed"])
            rows[key] = r["eval"]
            for c2, z in (r.get("zero_shot") or {}).items():
                rows[("zero_shot", r["arm"], c2, r["seed"])] = z
    return rows

def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, len(vals)
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), len(vals)

def main():
    rows = load_runs()
    groups = defaultdict(lambda: defaultdict(list))
    for (kind, arm, cond, seed), ev in rows.items():
        for b in BUCKETS:
            if b in ev["buckets"]:
                groups[(kind, arm, cond)][b].append(ev["buckets"][b])
        groups[(kind, arm, cond)]["overall"].append(ev["overall"])

    table = {}
    for key, d in groups.items():
        table["|".join(key)] = {b: agg(d.get(b, [])) for b in BUCKETS + ["overall"]}

    # CSV: one row per arm/cond/seed + per bucket
    with open(OUT / "e4_full.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["kind", "arm", "cond", "seed"] + [f"acc_{b}" for b in BUCKETS] + ["acc_overall"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for (kind, arm, cond, seed), ev in sorted(rows.items()):
            row = {"kind": kind, "arm": arm, "cond": cond, "seed": seed,
                   "acc_overall": ev["overall"]}
            for b in BUCKETS:
                row[f"acc_{b}"] = ev["buckets"].get(b, "")
            w.writerow(row)

    merged = {"groups": {k: {b: {"mean": v[0], "sd": v[1], "n": v[2]}
                             for b, v in d.items()} for k, d in table.items()},
              "raw_runs": len(rows)}
    (OUT / "e4_full_merged.json").write_text(json.dumps(merged, indent=1), encoding="utf-8")

    def cell(key, b):
        v = table.get(key, {}).get(b)
        if not v or v[0] is None:
            return "—"
        return f"{v[0]:.3f}±{v[1]:.3f}" if v[2] > 1 else f"{v[0]:.3f}"

    def col(key):
        return [cell(key, b) for b in BUCKETS] + [cell(key, "overall")]

    order = [("trained","cross","real"), ("trained","cross","shuffle"),
             ("trained","cross","nomarker"), ("trained","cross","pos"),
             ("zero_shot","cross","shuffle"), ("zero_shot","cross","nomarker"),
             ("trained","fir16","real"), ("trained","gru128","real"),
             ("zero_shot","fir16","shuffle"), ("zero_shot","fir16","nomarker"),
             ("zero_shot","gru128","shuffle"), ("zero_shot","gru128","nomarker")]

    lines = ["# E4 full (T=96, 4000 steps, 3 seeds)", "",
             "| arm/cond | 1–8 | 9–16 | 17–32 | 33–64 | overall |",
             "|---|---:|---:|---:|---:|---:|"]
    for k in order:
        key = "|".join(k)
        if key not in table:
            continue
        name = f"{k[1]}/{k[2]}" + (" (zs)" if k[0] == "zero_shot" else "")
        lines.append("| " + name + " | " + " | ".join(col(key)) + " |")

    # verdict per pre-registration
    def mean_of(arm, cond, b, kind="trained"):
        v = table.get(f"{kind}|{arm}|{cond}", {}).get(b)
        return v[0] if v and v[0] is not None else None

    def max_seed(arm, cond, b, kind="trained"):
        vals = groups.get((kind, arm, cond), {}).get(b, [])
        return max(vals) if vals else None

    cross_long = [mean_of("cross","real",b) for b in ("17-32","33-64")]
    fir_long = [max_seed("fir16","real",b) for b in ("17-32","33-64")]
    gru_long = [max_seed("gru128","real",b) for b in ("17-32","33-64")]
    sh = mean_of("cross","shuffle","overall","zero_shot")
    nm = mean_of("cross","nomarker","overall","zero_shot")
    pos = mean_of("cross","pos","overall")
    c1 = all(v is not None and v >= .85 for v in cross_long)
    c2 = all(v is not None and v <= .55 for v in fir_long + gru_long)
    c3 = all(v is not None and .45 <= v <= .55 for v in (sh, nm))
    c4 = pos is not None and pos >= .90
    ok = c1 and c2 and c3 and c4

    lines += ["", "## 预注册判定", "",
              f"- cross 17–32/33–64 ≥.85: {c1} ({cross_long})",
              f"- fir16/gru 17–32/33–64 ≤.55: {c2} (fir={fir_long}, gru={gru_long})",
              f"- shuffle/nomarker ≈.5: {c3} ({sh:.3f}, {nm:.3f})",
              f"- pos ≥.9: {c4} ({pos})",
              f"- **学到的内容寻址: {'成立' if ok else '不成立'}**", "",
              "说明：",
              "- cross 三个 seed 在 17–32/33–64 全 1.000，shuffle/nomarker 贴 chance，pos 1.000。",
              "- 但 gru128 也在 17–32/33–64 全 1.000（3/3 seed），不满足预注册的 GRU ≤.55 对照；",
              "  该任务只含一个 marker，GRU 可用单槽 latch 学出内容寻址，因此本实验不能把",
              "  “任意 lag 回忆”归因于显式 cross-attn 指针。按预注册规则判定为不成立。",
              "- fir16 窗外仍失败（17–32 max .516、33–64 max .549），窗口限制成立。"]
    (OUT / "e4_full_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()