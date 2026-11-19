# m2b_ctx.py — 上下文缩放实验（T=192 主实验 / T=96 对照），基于 m2b_causal.py
# 只改：T、B、cap/min/patience、臂集合。数据/STAMP/splits/h-only 头均不动。
# 臂：bypass（写在过去位置，需注意力回看 lag 步）/ shift（oracle，写在当前位置）/ gru（因果摘要）
# CPU 冒烟：M2B_SMOKE=1 -> T=64, B=2, 150 步, 仅 bypass@15
import sys, os, time, copy, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

ROOT = Path(r"E:\rwkv7"); sys.path.insert(0, str(ROOT / "kronos_src"))
from model import Kronos

def E(k, d): return os.environ.get("M2B_" + k, str(d))

DEV = E("DEV", "cuda")
if DEV.startswith("cuda") and not torch.cuda.is_available(): DEV = "cpu"
OUT = Path(E("OUT", r"C:\Users\lele\AppData\Local\Temp\opencode\m2b_ctx"))
OUT.mkdir(parents=True, exist_ok=True)
SMOKE = E("SMOKE", "0") == "1"

if SMOKE:
    d = dict(T=64, B=2, CAP=150, MIN=150, PATIENCE=10 ** 9, EVAL_EVERY=50,
             EVAL_B=4, NV=2, NT=2, SEEDS="0", LAGS="15", ARMS="bypass", SAVE_STATE=0)
else:
    d = dict(T=192, B=16, CAP=8000, MIN=2000, PATIENCE=800, EVAL_EVERY=200,
             EVAL_B=32, NV=8, NT=16, SEEDS="0", LAGS="15,30",
             ARMS="bypass,shift,gru", SAVE_STATE=0)
cfg = {}
for k, v in d.items():
    if k in ("SEEDS", "LAGS", "ARMS"):
        cfg[k] = [x.strip() for x in E(k, v).split(",") if x.strip()]
        if k in ("SEEDS", "LAGS"): cfg[k] = [int(x) for x in cfg[k]]
    else:
        cfg[k] = type(v)(E(k, v))
cfg["OUT"] = OUT; cfg["DEV"] = DEV; cfg["SMOKE"] = SMOKE
T, B = cfg["T"], cfg["B"]
for lag in cfg["LAGS"]:
    assert 0 < lag < T - 1, f"lag {lag} 与 T {T} 不兼容"
print(json.dumps({"cfg": {k: str(v) for k, v in cfg.items()}}, ensure_ascii=False), flush=True)

df = pd.read_csv(ROOT / "data" / "binance" / "klines_1m" / "bars_1m_of_3m.csv", usecols=["ts"])
N = len(df); i1, i2 = int(N * 0.70), int(N * 0.85)
ts = pd.to_datetime(df["ts"], unit="s", utc=True)
z = np.load(ROOT / "runs" / "e6_long_2026-10-20" / "e5_tokens.npz")
STAMP = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1).astype(np.float32)
S1t = torch.tensor(z["S1"]); S2t = torch.tensor(z["S2"]); STt = torch.tensor(STAMP)
print(f"N={N} train<{i1} val<{i2} test<{N}  T={T} B={B} tokens/step={B*T}", flush=True)

model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(DEV).eval()
for p in model.parameters(): p.requires_grad = False
H = model.d_model; NL = len(model.transformer); FF = model.ff_dim

def mem_est(T_, B_):
    """粗略显存估算（fp32）：前向激活 + 反传峰值 + 权重 + CUDA 上下文。"""
    fwd = B_ * T_ * NL * (8 * H + 3 * FF) * 4
    attn = B_ * model.n_heads * T_ * T_ * 4 * 2  # 若 SDPA 物化 scores/softmax 的上限
    w = sum(p.numel() for p in model.parameters()) * 4
    peak = 2.5 * fwd + attn + w + 0.5 * 2 ** 30
    g = lambda b: round(b / 2 ** 30, 2)
    return {"T": T_, "B": B_, "tokens": B_ * T_, "fwd_act_gb": g(fwd),
            "attn_upper_gb": g(attn), "peak_est_gb": g(peak)}

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
        B_, T_, _ = x.shape
        if s.arm == "bypass":
            mem = m
        elif s.arm == "shift":
            mem = torch.zeros_like(m); mem[:, lag:] = m[:, :T_ - lag]
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

def batch(lo, hi, B_, g):
    st = g.integers(lo, hi - T, B_); idx = torch.tensor(st[:, None] + np.arange(T))
    c = torch.tensor(g.choice([-1.0, 1.0], size=(B_, T)).astype(np.float32))
    zz = torch.tensor(g.standard_normal((B_, T)).astype(np.float32))
    return (S1t[idx].to(DEV), S2t[idx].to(DEV), STt[idx].to(DEV),
            torch.stack([c, zz], -1).to(DEV), c.to(DEV), st)

def make_sets(lag, vg, tg):
    def one(lo, hi, g):
        s1, s2, st, m, c, pos = batch(lo, hi, cfg["EVAL_B"], g)
        return s1, s2, st, m, (c[:, :T - lag] > 0).float(), pos
    return ([one(i1, i2, vg) for _ in range(cfg["NV"])],
            [one(i2, N, tg) for _ in range(cfg["NT"])])

def evaluate(net, S, lag):
    net.eval(); cor = n = 0
    with torch.no_grad():
        for s1, s2, st, m, y, _ in S:
            lg = net(s1, s2, st, m, lag)[:, lag:]
            cor += ((lg > 0).float() == y).sum().item(); n += y.numel()
    net.train(); return cor / n

def smoke_checks():
    """形状 / 位置对齐 / 内存估算，不参与训练。"""
    print("\n===== CPU 冒烟检查 =====", flush=True)
    print(f"序列长度 T={T}，batch B={B}，每步 token={B*T}", flush=True)
    g = np.random.default_rng(0)
    s1, s2, st, m, c, pos = batch(0, i1, B, g)
    print(f"首窗起点 pos={pos.tolist()}，窗口末端 max={int(pos.max())+T} <= i1={i1}? "
          f"{int(pos.max())+T <= i1}", flush=True)
    lag = cfg["LAGS"][0]
    y = (c[:, :T - lag] > 0).float()
    # 标签：位置 p>=lag 的目标是 c[p-lag]；与 y 索引一致
    assert torch.equal(y, (c[:, :T - lag] > 0).float()) and y.shape[1] == T - lag
    assert torch.equal(y[0, :8], (c[0, :8] > 0).float())
    print(f"标签对齐: y 形状 {tuple(y.shape)} = T-lag；位置 p={lag}..{T-1} 的标签取自 c[p-{lag}]", flush=True)
    # shift 注入位置
    mem = torch.zeros_like(m); mem[:, lag:] = m[:, :T - lag]
    assert mem[:, :lag].abs().sum() == 0 and torch.equal(mem[:, lag:], m[:, :T - lag])
    print(f"shift 注入: mem[p] = m[p-{lag}]（p<{lag} 置零）确认", flush=True)
    x = model.embedding([s1, s2]) + model.time_emb(st)
    print(f"embedding+time_emb 形状 {tuple(x.shape)}，d_model={H}，depth={NL}，ff={FF}", flush=True)
    net = Net(cfg["ARMS"][0]).to(DEV)
    lg = net(s1, s2, st, m, lag)
    assert lg.shape == (B, T) and lg[:, lag:].shape[1] == y.shape[1]
    print(f"logits 形状 {tuple(lg.shape)}，取 [:, {lag}:] -> {tuple(lg[:, lag:].shape)}，与 y 匹配", flush=True)
    print(f"可训练参数 {sum(p.numel() for p in net.parameters()):,}", flush=True)
    print("\n显存估算（fp32，SDPA 物化上界；CUDA 上下文另计 0.5GB）:", flush=True)
    for row in [mem_est(96, 32), mem_est(192, 16), mem_est(T, B)]:
        print(f"  {json.dumps(row)}", flush=True)
    print("结论：T=192/B=16 与 T=96/B=32 的 token 数相同，激活量同量级；"
          "若 OOM 先降 B 到 8。\n", flush=True)

if SMOKE: smoke_checks()

def run(arm, lag, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Net(arm).to(DEV)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    VAL, TEST = make_sets(lag, np.random.default_rng(999), np.random.default_rng(777))
    best = -1.0; best_state = None; best_step = 0; t0 = time.time(); it = 0
    for it in range(1, cfg["CAP"] + 1):
        s1, s2, st, m, c, _ = batch(0, i1, B, rng)
        y = (c[:, :T - lag] > 0).float()
        lg = net(s1, s2, st, m, lag)[:, lag:]
        loss = lossf(lg, y); opt.zero_grad(); loss.backward(); opt.step()
        if it % cfg["EVAL_EVERY"] == 0:
            acc = evaluate(net, VAL, lag)
            if acc > best + 1e-3:
                best, best_step = acc, it
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            if it >= cfg["MIN"] and it - best_step >= cfg["PATIENCE"]:
                print(f"  early-stop {arm}@{lag} s{seed} it={it} best_step={best_step}", flush=True)
                break
            if SMOKE or cfg["EVAL_EVERY"] >= 1000:
                print(f"  [v] {arm}@{lag} s{seed} it={it} loss={loss.item():.4f} val={acc:.4f} best={best:.4f}", flush=True)
        if it % max(1, cfg["EVAL_EVERY"] * 5) == 0 and not SMOKE:
            print(f"  {arm}@{lag} s{seed} it={it} loss={loss.item():.4f} val={best:.4f}", flush=True)
    last_test = evaluate(net, TEST, lag)
    net.load_state_dict(best_state); best_test = evaluate(net, TEST, lag)
    sp = None
    if cfg["SAVE_STATE"]:
        d = OUT / "states"; d.mkdir(exist_ok=True); sp = str(d / f"{arm}_l{lag}_s{seed}.pt")
        torch.save(net.state_dict(), sp)
    return {"arm": arm, "lag": lag, "seed": seed, "T": T, "B": B,
            "val_best": round(best, 4), "test_best": round(best_test, 4),
            "test_last": round(last_test, 4), "gap": round(best - best_test, 4),
            "best_step": best_step, "steps": it,
            "params": sum(p.numel() for p in net.parameters()), "s": round(time.time() - t0, 1)}

rows = []
J = OUT / ("smoke_results.jsonl" if SMOKE else "results.jsonl"); J.write_text("", encoding="utf-8")
for lag in cfg["LAGS"]:
    for arm in cfg["ARMS"]:
        for seed in cfg["SEEDS"]:
            r = run(arm, lag, seed); rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
            with J.open("a", encoding="utf-8") as f: f.write(json.dumps(r, ensure_ascii=False) + "\n")

d = pd.DataFrame(rows)
print("\n===== 明细 ====="); print(d.to_string(index=False), flush=True)
print("\n===== 判定（T=%d） =====" % T, flush=True)
for lag in cfg["LAGS"]:
    b = d[(d.arm == "bypass") & (d.lag == lag)].test_best
    s = d[(d.arm == "shift") & (d.lag == lag)].test_best
    gp = d[(d.arm == "bypass") & (d.lag == lag)].gap
    if len(b) == 0: continue
    print(f"L={lag}: bypass test_best={b.mean():.4f} last="
          f"{d[(d.arm=='bypass')&(d.lag==lag)].test_last.mean():.4f} gap={gp.mean():.4f}"
          + (f" | shift={s.mean():.4f}" if len(s) else ""), flush=True)
d.to_csv(OUT / ("smoke_results.csv" if SMOKE else "results.csv"), index=False, encoding="utf-8")
if not SMOKE:
    b15 = d[(d.arm == "bypass") & (d.lag == 15)].test_best
    b30 = d[(d.arm == "bypass") & (d.lag == 30)].test_best
    if T >= 180 and len(b15):
        v = float(b15.mean())
        if v > 0.60:
            verdict = "支持 R∝T：L15 墙是相对的（R≈22>15），但 L30 应仍≈0.5"
        elif v < 0.55:
            verdict = "不支持 R∝T：T=192 下 L15 仍≈0.51，墙不是上下文长度能解决的"
        else:
            verdict = "灰区（0.55~0.60），需补 seed 或 T=384 再判"
        print(f"\n>>> 判定: bypass L15={v:.4f} -> {verdict}", flush=True)
        if len(b30):
            print(f">>> 参照: bypass L30={b30.mean():.4f}（预期≈0.5，若随 T 一起升高则不只是 R 的问题）", flush=True)
    elif T < 180 and len(b15):
        print(f"\n>>> 对照(T={T}) bypass L15={b15.mean():.4f}（m2b_causal 基线 .5136；±0.03 内视为复现）", flush=True)
print(f"\n结果写入 {OUT}，总耗时 {d.s.sum():.0f}s", flush=True)