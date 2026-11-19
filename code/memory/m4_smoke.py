# M4 smoke (CPU): 只验证管线 + 因果位置，不做全量训练。
# 与 m4_append.py 的差异：dev=cpu, T=32, B=4, cap=100(env M4_CAP), lags=[5,10,30],
# 训练/验证 batch 缩小, 每 25 步评估一次, 防 cap<100 时 best_state=None 崩溃。
import sys, time, copy, json, os
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

ROOT = Path(r"E:\rwkv7"); sys.path.insert(0, str(ROOT / "kronos_src"))
from model import Kronos

dev = "cpu"
torch.set_num_threads(int(os.environ.get("M4_THREADS", "4")))
T = int(os.environ.get("M4_T", "32"))
B = int(os.environ.get("M4_B", "4"))
CAP = int(os.environ.get("M4_CAP", "100"))
LAGS = [int(x) for x in os.environ.get("M4_LAGS", "5,10,30").split(",")]
VB, TB = int(os.environ.get("M4_VB", "8")), int(os.environ.get("M4_TB", "8"))

df = pd.read_csv(ROOT / "data" / "binance" / "klines_1m" / "bars_1m_of_3m.csv", usecols=["ts"])
N = len(df); i1, i2 = int(N * 0.70), int(N * 0.85)
z = np.load(ROOT / "runs" / "e6_long_2026-10-20" / "e5_tokens.npz")
ts = pd.to_datetime(df["ts"], unit="s", utc=True)
STAMP = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1).astype(np.float32)
S1t = torch.tensor(z["S1"], device=dev); S2t = torch.tensor(z["S2"], device=dev); STt = torch.tensor(STAMP, device=dev)

model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
for p in model.parameters(): p.requires_grad = False
H = model.d_model

def mlp(i, w, o): return nn.Sequential(nn.Linear(i, w), nn.GELU(), nn.Linear(w, o))

class AppendNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc_m = mlp(2, 64, H)
        s.enc_head = mlp(2, 64, H)
        s.pi = nn.Sequential(nn.Linear(2 * H, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(s, s1, s2, st, m):
        xb = model.embedding([s1, s2]) + model.time_emb(st)
        xm = s.enc_m(m)
        x = torch.stack([xm, xb], dim=2).reshape(s1.shape[0], 2 * T, H)   # [m_t, bar_t] 交替
        for layer in model.transformer:
            x = layer(x)
        x = model.norm(x)
        hb = x[:, 1::2, :]                                                # bar 位置
        return s.pi(torch.cat([hb / (hb.norm(dim=-1, keepdim=True) + 1e-6) * 8, s.enc_head(m)], -1)).squeeze(-1)

def batch(lo, hi, b, g):
    st = g.integers(lo, hi - T, b); idx = torch.tensor(st[:, None] + np.arange(T), device=dev)
    c = torch.tensor(g.choice([-1.0, 1.0], size=(b, T)).astype(np.float32), device=dev)
    zz = torch.tensor(g.standard_normal((b, T)).astype(np.float32), device=dev)
    return S1t[idx], S2t[idx], STt[idx], torch.stack([c, zz], -1), c

def make_sets(lag, vg, tg):
    def one(lo, hi, b, g):
        s1, s2, st, m, c = batch(lo, hi, b, g)
        return s1, s2, st, m, (c[:, :T - lag] > 0).float()
    return [one(i1, i2, VB, vg) for _ in range(2)], [one(i2, N, TB, tg) for _ in range(3)]

def evaluate(net, S, lag):
    net.eval(); cor = n = 0
    with torch.no_grad():
        for s1, s2, st, m, y in S:
            lg = net(s1, s2, st, m)[:, lag:]
            cor += ((lg > 0).float() == y).sum().item(); n += y.numel()
    net.train(); return cor / n

def run(lag, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = AppendNet().to(dev)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    vg, tg = np.random.default_rng(999), np.random.default_rng(777)
    VAL, TEST = make_sets(lag, vg, tg)
    best, best_state = evaluate(net, VAL, lag), copy.deepcopy(net.state_dict())
    best_step = 0; t0 = time.time(); it = 0
    for it in range(1, CAP + 1):
        s1, s2, st, m, c = batch(0, i1, B, rng)
        y = (c[:, :T - lag] > 0).float()
        lg = net(s1, s2, st, m)[:, lag:]
        loss = lossf(lg, y); opt.zero_grad(); loss.backward(); opt.step()
        if it % 25 == 0:
            acc = evaluate(net, VAL, lag)
            if acc > best + 1e-3: best, best_state, best_step = acc, copy.deepcopy(net.state_dict()), it
    net.load_state_dict(best_state); test = evaluate(net, TEST, lag)
    return {"arm": "append", "lag": lag, "val": round(best, 4), "test": round(test, 4),
            "best_step": best_step, "steps": it,
            "params": sum(p.numel() for p in net.parameters()), "s": round(time.time() - t0, 1)}

if __name__ == "__main__":
    rows = []
    for lag in LAGS:
        r = run(lag); rows.append(r); print(json.dumps(r), flush=True)
    print(pd.DataFrame(rows)[["arm", "lag", "val", "test", "best_step", "steps", "params", "s"]].to_string(index=False))
    bad = [r for r in rows if r["lag"] == 5 and r["val"] < 0.52]
    if bad: print("WARN: L=5 val<0.52 -> 检查 enc_m / 位置对齐", flush=True)