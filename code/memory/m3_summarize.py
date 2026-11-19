import json
from pathlib import Path

d = Path(r"C:\Users\lele\AppData\Local\Temp\opencode\m3_crossmem")
rows = []
for p in sorted(d.glob("m3_v2smoke_*.json")):
    r = json.load(open(p, encoding="utf-8"))
    row = {"arm": r["arm"], "L": r["lag"], "params": r["trainable_params"],
           "val": round(r["val_acc_best"], 3), "test": round(r["test_acc"], 3)}
    for phase in ("addr_probe_init", "addr_probe_final"):
        pr = r.get(phase) or {}
        row[phase] = {L: {k: round(v, 3) for k, v in pr[L].items()} for L in pr}
    rows.append(row)
for r in rows:
    print(r["arm"], "L", r["L"], "p", r["params"], "val", r["val"], "test", r["test"])
    for phase in ("addr_probe_init", "addr_probe_final"):
        pr = r[phase]
        if not pr:
            continue
        print("  ", phase, {L: (pr[L]["addr_ratio"], pr[L]["addr_ratio_all"], pr[L]["peak_ratio"]) for L in pr})
print("JSON", d / "v2_smoke_table.json")
(d / "v2_smoke_table.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")