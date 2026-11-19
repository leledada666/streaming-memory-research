# M2b: 读路径的因果分解（替代 M4 的 append 对照）
# 同一个冻结 Kronos、同一套注入参数、同一个 h-only 决策头，只改"把哪一段 m 写到当前位置"：
#   bypass        : x_s += enc(m_s)          （写在过去位置，需要注意力回看）
#   shift         : x_t += enc(m_{t-L})      （oracle：把目标写到当前位置，局部可读）
#   shift_random  : x_t += enc(m_{t-Lr})     （局部可读但内容错，排除"任意局部注入"）
#   gru           : x_t += enc(GRU(m_<=t))   （因果摘要，局部可读）
#   none          : 不注入
# 若 shift 高、bypass 崩、shift_random 随机 => 瓶颈是"读过去"，不是容量/优化。
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

def mlp(i, w, o): return nn.Sequential(nn.Linear(i, w), nn.GELU(), nn.Linear(w, o))

class Net(nn.Module):
    def __init__(s, arm):
        super().__init__(); s.arm = arm
        s.gru = nn.GRU(2, 128, batch_first=True) if arm == "gru" else None
        in_dim = 128 if arm == "gru" else 2
        s.enc = nn.ModuleList([mlp(in_dim, 64, H) for _ in range(NL)])
        s.g = nn.Parameter(torch.full((NL,), -2.0))
        s.pi = nn.Sequential(nn.Linear(H, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(s, s1, s2, st, m, lag):
        x = model.embedding([s1, s2]) + model.time_emb(st)
        B, T_, _ = x.shape
        if s.arm == "bypass":
            mem = m
        elif s.arm == "shift":
            mem = torch.zeros_like(m); mem[:, lag:] = m[:, :T_-lag]
        elif s.arm == "shift_random":
            lr = max(1, lag // 2)
            mem = torch.zeros_like(m); mem[:, lr:] = m[:, :T_-lr]
        elif s.arm == "gru":
            mem, _ = s.gru(m)
        else:
            mem = torch.zeros_like(m)
        for i, layer in enumerate(model.transformer):
            u = s.enc[i](mem); u = u / (u.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)
            x = x + torch.sigmoid(s.g[i]) * u
            x = layer(x)
        h = model.norm(x)
        return s.pi(h / (h.norm(dim=-1, keepdim=True) + 1e-6) * 8).squeeze(-1)

def batch(lo, hi, B, g):
    st = g.integers(lo, hi - T, B); idx = torch.tensor(st[:, None] + np.arange(T), device=dev)
    c = torch.tensor(g.choice([-1.0, 1.0], size=(B, T)).astype(np.float32), device=dev)
    zz = torch.tensor(g.standard_normal((B, T)).astype(np.float32), device=dev)
    return S1t[idx], S2t[idx], STt[idx], torch.stack([c, zz], -1), c

def make_sets(lag, vg, tg, nv=8, nt=16):
    def one(lo, hi, B, g):
        s1, s2, st, m, c = batch(lo, hi, B, g)
        return s1, s2, st, m, (c[:, :T-lag] > 0).float()
    return [one(i1, i2, 64, vg) for _ in range(nv)], [one(i2, N, 64, tg) for _ in range(nt)]

def evaluate(net, S, lag):
    net.eval(); cor = n = 0
    with torch.no_grad():
        for s1, s2, st, m, y in S:
            lg = net(s1, s2, st, m, lag)[:, lag:]
            cor += ((lg > 0).float() == y).sum().item(); n += y.numel()
    net.train(); return cor/n

def run(arm, lag, seed=0, cap=int(os.environ.get("M2B_CAP", "15000")), min_steps=2000, patience=800):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Net(arm).to(dev); opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    vg, tg = np.random.default_rng(999), np.random.default_rng(777)
    VAL, TEST = make_sets(lag, vg, tg)
    best = -1; best_state = None; best_step = 0; t0 = time.time(); it0 = 0
    ck = None
    if os.environ.get("M2B_CKPT"):
        ck = Path(os.environ["M2B_CKPT"]) / f"m2b_{arm}_{lag}.pt"
        if ck.exists() and os.environ.get("M2B_RESUME"):
            d = torch.load(ck, weights_only=False)
            net.load_state_dict(d["net"]); opt.load_state_dict(d["opt"])
            best, best_state, best_step, it0 = d["best"], d["best_state"], d["best_step"], d["it"]
            torch.set_rng_state(d["trng"]); rng.bit_generator.state = d["nrng"]
            t0 = time.time() - d["elapsed"]; print(f"  resume {arm}@{lag} from it={it0}", flush=True)
    for it in range(it0+1, cap+1):
        s1, s2, st, m, c = batch(0, i1, 32, rng)
        y = (c[:, :T-lag] > 0).float()
        lg = net(s1, s2, st, m, lag)[:, lag:]
        loss = lossf(lg, y); opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 0:
            acc = evaluate(net, VAL, lag)
            if os.environ.get("M2B_VERBOSE"):
                print(f"  [{arm}@{lag}] it={it} val={acc:.4f} best={best:.4f}@{best_step}", flush=True)
            if acc > best + 1e-3: best, best_state, best_step = acc, copy.deepcopy(net.state_dict()), it
            if it >= min_steps and it - best_step >= patience: break
            if ck is not None:
                torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "best": best,
                            "best_state": best_state, "best_step": best_step, "it": it,
                            "trng": torch.get_rng_state(), "nrng": rng.bit_generator.state,
                            "elapsed": time.time()-t0}, ck)
    last_test = evaluate(net, TEST, lag)
    net.load_state_dict(best_state); best_test = evaluate(net, TEST, lag)
    return {"arm": arm, "lag": lag, "seed": seed, "val_best": round(best,4), "test_best": round(best_test,4),
            "test_last": round(last_test,4), "best_step": best_step, "steps": it,
            "params": sum(p.numel() for p in net.parameters()), "s": round(time.time()-t0,1)}

RUNS = [(a, l) for l in [0, 10, 15, 30, 60] for a in ["bypass", "shift", "shift_random", "gru", "none"]]
# 优先级：先跑 lag=15 的五个臂，再做其他 lag；可用 M2B_ONLY 环境变量过滤
only = os.environ.get("M2B_ONLY", "")
if only:
    RUNS = [(a, l) for (a, l) in RUNS if f"{a}@{l}" in only.split(",")]
rows = []
for arm, lag in RUNS:
    r = run(arm, lag); rows.append(r); print(json.dumps(r), flush=True)
print(pd.DataFrame(rows)[["arm","lag","val_best","test_best","test_last","best_step","steps","params","s"]].to_string(index=False))