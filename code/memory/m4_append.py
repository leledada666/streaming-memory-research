# M4: 关键对照——把 m_t 作为"新 token"append 到过去位置，能否突破 L 墙？
# 序列变成 [m_t, bar_t] 交替，长度 2T；bar_t 的 query 要读 m_{t-L}，距离 = 2L+1。
# 若冻结注意力是局部的，append 应比 bypass 更早崩（距离翻倍），证明瓶颈是"读"而非"接口"。
import sys, time, copy, json, os
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

ROOT = Path(r"E:\rwkv7"); sys.path.insert(0, str(ROOT / "kronos_src"))
from model import Kronos
dev = "cuda"

df = pd.read_csv(ROOT / "data" / "binance" / "klines_1m" / "bars_1m_of_3m.csv", usecols=["ts"])
N = len(df); i1, i2 = int(N*0.70), int(N*0.85)
z = np.load(ROOT / "runs" / "e6_long_2026-10-20" / "e5_tokens.npz")
ts = pd.to_datetime(df["ts"], unit="s", utc=True)
STAMP = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1).astype(np.float32)
S1t = torch.tensor(z["S1"], device=dev); S2t = torch.tensor(z["S2"], device=dev); STt = torch.tensor(STAMP, device=dev)

model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
for p in model.parameters(): p.requires_grad = False
H = model.d_model; T = 96

def mlp(i, w, o): return nn.Sequential(nn.Linear(i, w), nn.GELU(), nn.Linear(w, o))

class AppendNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc_m = mlp(2, 64, H)      # m_t -> token
        s.enc_head = mlp(2, 64, H)   # 当前 m 直达决策头（与 E6 对齐）
        s.pi = nn.Sequential(nn.Linear(2*H, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(s, s1, s2, st, m):
        xb = model.embedding([s1, s2]) + model.time_emb(st)        # B,T,H
        xm = s.enc_m(m)                                            # B,T,H
        x = torch.stack([xm, xb], dim=2).reshape(s1.shape[0], 2*T, H)  # [m_t, bar_t] 交替
        for layer in model.transformer:
            x = layer(x)
        x = model.norm(x)
        hb = x[:, 1::2, :]                                         # bar 位置
        return s.pi(torch.cat([hb / (hb.norm(dim=-1, keepdim=True) + 1e-6) * 8, s.enc_head(m)], -1)).squeeze(-1)

def batch(lo, hi, B, g):
    st = g.integers(lo, hi - T, B); idx = torch.tensor(st[:, None] + np.arange(T), device=dev)
    c = torch.tensor(g.choice([-1.0, 1.0], size=(B, T)).astype(np.float32), device=dev)
    zz = torch.tensor(g.standard_normal((B, T)).astype(np.float32), device=dev)
    return S1t[idx], S2t[idx], STt[idx], torch.stack([c, zz], -1), c

def make_sets(lag, vg, tg):
    def one(lo, hi, B, g):
        s1, s2, st, m, c = batch(lo, hi, B, g)
        return s1, s2, st, m, (c[:, :T-lag] > 0).float()
    return [one(i1, i2, 64, vg) for _ in range(4)], [one(i2, N, 64, tg) for _ in range(8)]

def evaluate(net, S, lag):
    net.eval(); cor = n = 0
    with torch.no_grad():
        for s1, s2, st, m, y in S:
            lg = net(s1, s2, st, m)[:, lag:]
            cor += ((lg > 0).float() == y).sum().item(); n += y.numel()
    net.train(); return cor/n

def run(lag, seed, cap=int(os.environ.get("M4_CAP", "2500"))):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = AppendNet().to(dev); opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    vg, tg = np.random.default_rng(999), np.random.default_rng(777)
    VAL, TEST = make_sets(lag, vg, tg)
    best = -1; best_state = None; best_step = 0; t0 = time.time()
    for it in range(1, cap+1):
        s1, s2, st, m, c = batch(0, i1, 32, rng)
        y = (c[:, :T-lag] > 0).float()
        lg = net(s1, s2, st, m)[:, lag:]
        loss = lossf(lg, y); opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0:
            acc = evaluate(net, VAL, lag)
            if acc > best + 1e-3: best, best_state, best_step = acc, copy.deepcopy(net.state_dict()), it
            if it >= 1000 and it - best_step >= 600: break
    net.load_state_dict(best_state); test = evaluate(net, TEST, lag)
    return {"arm": "append", "lag": lag, "seed": seed, "val": round(best,4), "test": round(test,4),
            "best_step": best_step, "steps": it, "params": sum(p.numel() for p in net.parameters()), "s": round(time.time()-t0,1)}

rows = []
for lag in [5, 10, 15, 30, 60]:
    r = run(lag, 0); rows.append(r); print(json.dumps(r), flush=True)
print(pd.DataFrame(rows)[["arm","lag","val","test","best_step","steps","params","s"]].to_string(index=False))