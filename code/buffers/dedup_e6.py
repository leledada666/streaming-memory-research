import glob, json, os
import pandas as pd

files = []
files += glob.glob(r"E:\rwkv7\runs\e6_long_2026-10-20\e6long_*.json")
files += glob.glob(r"E:\rwkv7\runs\overnight_2026-10-18\e6_*.json")
# 也收 4k 的 e6 结果（若在别处）
files += glob.glob(r"E:\rwkv7\runs\overnight_2026-10-18\e5*.json")
files = sorted(set(files))
rows = []
for f in files:
    try:
        j = json.load(open(f, encoding="utf-8"))
    except Exception as e:
        print("skip", f, e); continue
    v = j.get("variant") or j.get("arm") or "?"
    rows.append({
        "file": os.path.basename(f), "variant": v, "lag": j.get("lag"), "seed": j.get("seed"),
        "test": j.get("test_acc"), "val": j.get("best_val_acc") or j.get("val_acc_best"),
        "best_step": j.get("best_step"), "steps": j.get("steps_run") or j.get("steps"),
        "stop": j.get("stop_reason"), "params": j.get("trainable_params"),
    })
df = pd.DataFrame(rows)
print("files", len(files), "rows", len(df))
print("\n=== 原始 ===")
print(df.sort_values(["lag","variant","seed"])[["variant","lag","seed","test","val","best_step","steps","stop","file"]].to_string(index=False))

# 去重：同 (variant, lag, seed, test/val 四位小数)
df["key"] = df.apply(lambda r: f"{r.variant}|{r.lag}|{r.seed}|{round(float(r.test),4) if pd.notna(r.test) else 'NA'}|{round(float(r.val),4) if pd.notna(r.val) else 'NA'}", axis=1)
ded = df.drop_duplicates("key").copy()
print("\n=== 去重后 ===")
print(ded.sort_values(["lag","variant","seed"])[["variant","lag","seed","test","val","best_step","steps","stop","file"]].to_string(index=False))

print("\n=== 按 variant/lag 聚合（去重后）===")
if len(ded):
    g = ded.groupby(["variant","lag"]).agg(n=("seed","count"), test_mean=("test","mean"), test_min=("test","min"), test_max=("test","max"),
                                           val_mean=("val","mean"), seeds=("seed", lambda s: list(s))).reset_index()
    print(g.to_string(index=False))