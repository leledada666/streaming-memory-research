# M1: 冻结 Kronos-small 的注意力到底能回看多远？
# 零训练：在真实价格 token 上，测 query(t+L) -> key(t) 的注意力权重随 L 的衰减。
# 若 L>=15 时权重接近均匀分布，则 bypass 的记忆墙是"读不到"，不是"存不下"。
import math, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(r"E:\rwkv7")
sys.path.insert(0, str(ROOT / "kronos_src"))
from model import Kronos

dev = "cuda"
torch.manual_seed(0)
z = np.load(ROOT / "runs" / "e6_long_2026-10-20" / "e5_tokens.npz")
df = pd.read_csv(ROOT / "data" / "binance" / "klines_1m" / "bars_1m_of_3m.csv", usecols=["ts"])
ts = pd.to_datetime(df["ts"], unit="s", utc=True)
STAMP = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1).astype(np.float32)
N = len(df); T = 96; B = 32
rng = np.random.default_rng(123)
starts = rng.integers(int(N * 0.85), N - T, B)
idx = starts[:, None] + np.arange(T)
s1 = torch.tensor(z["S1"][idx], device=dev)
s2 = torch.tensor(z["S2"][idx], device=dev)
st = torch.tensor(STAMP[idx], device=dev)

model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
LAGS = [1, 2, 3, 5, 8, 10, 12, 15, 20, 30, 45, 60, 90]
records = []

def make_hook(layer_i):
    def hook(mod, inp, out):
        x = inp[0]
        b, t, _ = x.shape
        q = mod.q_proj(x).view(b, t, mod.n_heads, mod.head_dim).transpose(1, 2)
        k = mod.k_proj(x).view(b, t, mod.n_heads, mod.head_dim).transpose(1, 2)
        q, k = mod.rotary(q, k, position_offset=0)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(mod.head_dim)
        mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        att = logits.masked_fill(~mask, float("-inf")).softmax(-1)
        rec = {"layer": layer_i, "b": b, "t": t}
        for L in LAGS:
            if L >= t: continue
            d = torch.diagonal(att[:, :, L:, : t - L], dim1=-2, dim2=-1)  # B,H,n
            rec[f"L{L}"] = d.mean(dim=(0, 2)).detach().cpu().numpy()      # per head
        # 局部注意力质量：每个 query 落在最近 4 个 key 上的比例
        local = torch.zeros(b, t, device=x.device)
        for off in range(4):
            if off == 0:
                local += att[:, :, torch.arange(t), torch.arange(t)].mean(1)
            else:
                local[:, off:] += att[:, :, torch.arange(off, t), torch.arange(0, t - off)].mean(1)
        rec["local4"] = local.mean().item()
        ent = -(att.clamp_min(1e-9).log() * att).sum(-1).mean().item()
        rec["entropy"] = ent
        records.append(rec)
    return hook

hooks = [layer.self_attn.register_forward_hook(make_hook(i)) for i, layer in enumerate(model.transformer)]
with torch.no_grad():
    model.forward_hidden(s1, s2, st)
for h in hooks: h.remove()

# 按层聚合
rows = []
for L in LAGS:
    per_layer = {r["layer"]: r[f"L{L}"] for r in records if f"L{L}" in r}
    if not per_layer: continue
    t = T
    uni = np.mean([1.0 / (i + 1) for i in range(L, t)])
    overall = np.mean([v.mean() for v in per_layer.values()])
    top_head = np.mean([v.max() for v in per_layer.values()])
    rows.append({"L": L, "attn": round(float(overall), 6), "uniform": round(float(uni), 6),
                 "x_uniform": round(float(overall / uni), 3), "best_head": round(float(top_head), 6),
                 "best_head_x": round(float(top_head / uni), 3),
                 "per_layer": [round(float(v.mean()), 5) for v in per_layer.values()]})
print(pd.DataFrame(rows).to_string(index=False))
print("\nlocal4 mass per layer:", [round(r["local4"], 3) for r in records])
print("entropy per layer:", [round(r["entropy"], 3) for r in records])