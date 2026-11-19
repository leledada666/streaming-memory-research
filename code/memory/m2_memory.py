# M2: 记忆墙在"读"还是"存"？
# 同一冻结 Kronos + 同一合成任务 y_t = c_{t-L}，比较三种接口：
#   bypass      : 当前 m_t 经 gate*MLP 注入每层（E6 原版）
#   gru_bypass  : GRU(2->128) 生成当前摘要 s_t，再经 gate*MLP 注入每层
#   gru_head    : GRU 摘要只进决策头，完全不碰 backbone
# 预测：若墙在"读"，GRU 两臂在 L=15/30/60 仍高，bypass 崩。
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
H = model.d_model; NL = len(model.transformer); T = 96

def mlp(i, w, o):
    return nn.Sequential(nn.Linear(i, w), nn.GELU(), nn.Linear(w, o))

class Net(nn.Module):
    def __init__(s, arm):
        super().__init__(); s.arm = arm
        if arm == "bypass":
            s.enc = nn.ModuleList([mlp(2, 64, H) for _ in range(NL)]); s.g = nn.Parameter(torch.full((NL,), -2.0))
            s.enc_head = mlp(2, 64, H)
        elif arm == "gru_bypass":
            s.gru = nn.GRU(2, 128, batch_first=True)
            s.enc = nn.ModuleList([mlp(128, 64, H) for _ in range(NL)]); s.g = nn.Parameter(torch.full((NL,), -2.0))
            s.enc_head = mlp(128, 64, H)
        elif arm == "gru_head":
            s.gru = nn.GRU(2, 128, batch_first=True); s.enc_head = mlp(128, 64, H)
        elif arm == "none":
            s.enc_head = mlp(2, 64, H)
        s.pi = nn.Sequential(nn.Linear(2*H, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(s, s1, s2, st, m):
        x = model.embedding([s1, s2]) + model.time_emb(st)
        if s.arm == "bypass":
            mem = m
        elif s.arm in ("gru_bypass", "gru_head"):
            mem, _ = s.gru(m)
        else:
            mem = m
        for i, layer in enumerate(model.transformer):
            if s.arm in ("bypass", "gru_bypass"):
                u = s.enc[i](mem); u = u / (u.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)
                x = x + torch.sigmoid(s.g[i]) * u
            x = layer(x)
        h = model.norm(x); hm = s.enc_head(mem)
        if s.arm == "none": hm = torch.zeros_like(hm)
        return s.pi(torch.cat([h / (h.norm(dim=-1, keepdim=True) + 1e-6) * 8, hm], -1)).squeeze(-1)

def batch(lo, hi, B, g):
    st = g.integers(lo, hi - T, B); idx = torch.tensor(st[:, None] + np.arange(T), device=dev)
    c = torch.tensor(g.choice([-1.0, 1.0], size=(B, T)).astype(np.float32), device=dev)
    zz = torch.tensor(g.standard_normal((B, T)).astype(np.float32), device=dev)
    return S1t[idx], S2t[idx], STt[idx], torch.stack([c, zz], -1), c

def make_sets(lag, vg, tg):
    def one(lo, hi, B, g):
        s1, s2, st, m, c = batch(lo, hi, B, g)
        y = (c[:, :T-lag] > 0).float()
        return s1, s2, st, m, y
    return [one(i1, i2, 64, vg) for _ in range(4)], [one(i2, N, 64, tg) for _ in range(8)]

def evaluate(net, S, lag):
    net.eval(); cor = n = 0
    with torch.no_grad():
        for s1, s2, st, m, y in S:
            lg = net(s1, s2, st, m)[:, lag:]
            cor += ((lg > 0).float() == y).sum().item(); n += y.numel()
    net.train(); return cor/n

def run(arm, lag, seed, cap=int(os.environ.get("M2_CAP", "2500"))):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Net(arm).to(dev); opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    vg, tg = np.random.default_rng(999), np.random.default_rng(777)
    VAL, TEST = make_sets(lag, vg, tg)
    best = -1; best_state = None; best_step = 0; t0 = time.time()
    for it in range(1, cap+1):
        s1, s2, st, m, c = batch(0, i1, 32, rng)
        y = (c[:, :T-lag] > 0).float()
        lg = net(s1, s2, st, m)[:, lag:]
        loss = lossf(lg, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0:
            acc = evaluate(net, VAL, lag)
            if acc > best + 1e-3: best, best_state, best_step = acc, copy.deepcopy(net.state_dict()), it
            if it >= 1000 and it - best_step >= 600: break
    net.load_state_dict(best_state)
    test = evaluate(net, TEST, lag)
    return {"arm": arm, "lag": lag, "seed": seed, "val": round(best, 4), "test": round(test, 4),
            "best_step": best_step, "steps": it, "params": sum(p.numel() for p in net.parameters()), "s": round(time.time()-t0,1)}

rows = []
for lag in [10, 15, 30, 60]:
    for arm in ["bypass", "gru_bypass", "gru_head"]:
        r = run(arm, lag, 0); rows.append(r); print(json.dumps(r), flush=True)
    r = run("none", lag, 0); rows.append(r); print(json.dumps(r), flush=True)
print(pd.DataFrame(rows)[["arm","lag","val","test","best_step","steps","params","s"]].to_string(index=False))