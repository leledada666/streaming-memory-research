# -*- coding: utf-8 -*-
"""E3 (M12): 弱写落盘 -- 注入幅度 x 读出方式 的因果对照 (CPU, frozen Kronos-small)
注入: x_l <- x_l + c_t * w,  w = unit * a * mean_t||x_l||   (a = ||c*w||/||x||)
读出: 同一层 attention 输出 (best head linear ridge probe -> c_{t-L})
  natural : 原始冻结注意力
  force   : logits[t, t-L] += BETA (强制寻址到目标 key)
  wrong   : logits[t, t-(L//2)] += BETA (错误 lag, 排除"任意偏置")
  uniform : 因果均匀注意力 (完全不做寻址, 上界=平均池化)
正控: L=0 (目标在当前位置), force 应 ~1.0
网格: a in {0.004,0.01,0.02,1.0}, L in {15,60}, layer in {0,7}, 3 seeds
落盘: m12_e3.json
"""
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(r"E:\rwkv7"); sys.path.insert(0, str(ROOT / "kronos_src"))
from model import Kronos

dev = "cpu"; torch.set_num_threads(8)
OUT = Path(r"C:\Users\lele\AppData\Local\Temp\opencode\m12_e3.json")
T, B, NB = 96, 8, 4
AMPS = [0.004, 0.01, 0.02, 1.0]
LAGS = [15, 60]
LAYERS = [0, 7]
MODES = ["natural", "force", "wrong", "uniform"]
BETA = 40.0
SEEDS = [0, 1, 2]

df = pd.read_csv(ROOT / "data" / "binance" / "klines_1m" / "bars_1m_of_3m.csv", usecols=["ts"])
N = len(df)
z = np.load(ROOT / "runs" / "e6_long_2026-10-20" / "e5_tokens.npz")
ts = pd.to_datetime(df["ts"], unit="s", utc=True)
STAMP = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1).astype(np.float32)

model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
H = model.d_model
NH = model.transformer[0].self_attn.n_heads
DH = model.transformer[0].self_attn.head_dim


def make_batches(seed):
    rng = np.random.default_rng(100 + seed)
    out = []
    for _ in range(NB):
        st = rng.integers(int(N * 0.85), N - T, B)
        idx = torch.tensor(st[:, None] + np.arange(T))
        out.append((torch.tensor(z["S1"][idx.numpy()]).to(dev),
                    torch.tensor(z["S2"][idx.numpy()]).to(dev),
                    torch.tensor(STAMP[idx.numpy()]).to(dev),
                    torch.tensor(rng.choice([-1.0, 1.0], size=(B, T)).astype(np.float32)).to(dev)))
    return out


def layer_input(bi, li, batches):
    s1, s2, stamp, c = batches[bi]
    with torch.no_grad():
        x = model.embedding([s1, s2]) + model.time_emb(stamp)
        for j in range(li):
            x = model.transformer[j](x)
    return x


def head_out(xin, attn, norm1, mode, L, w, c):
    x = xin + c[:, :, None] * w[None, None, :]
    xn = norm1(x)
    b, t, _ = xn.shape
    q = attn.q_proj(xn).view(b, t, NH, DH).transpose(1, 2)
    k = attn.k_proj(xn).view(b, t, NH, DH).transpose(1, 2)
    v = attn.v_proj(xn).view(b, t, NH, DH).transpose(1, 2)
    q, k = attn.rotary(q, k, position_offset=0)
    mask = torch.tril(torch.ones(t, t, dtype=torch.bool))
    mass = float("nan")
    if mode == "uniform":
        att = mask.float() / mask.float().sum(-1, keepdim=True)
    else:
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(DH)
        if mode in ("force", "wrong"):
            lag = L if mode == "force" else (L // 2 if L // 2 >= 1 else 1)
            for tq in range(lag, t):
                logits[:, :, tq, tq - lag] += BETA
            rows = torch.arange(lag, t)
            mass = float(torch.softmax(logits.masked_fill(~mask, -1e30), -1)[
                :, :, rows, rows - lag].mean())
        att = logits.masked_fill(~mask, float("-inf")).softmax(-1)
    return att @ v, mass


def ridge_probe(Xtr, ytr, Xte, yte, lam=10.0):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    A = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]
    Bm = np.c_[np.ones(len(Xte)), (Xte - mu) / sd]
    M = A.T @ A + lam * np.eye(A.shape[1]); M[0, 0] -= lam
    beta = np.linalg.solve(M, A.T @ ytr)
    return float((np.sign(Bm @ beta) == yte).mean())


rows = []
t0 = time.time()
for seed in SEEDS:
    batches = make_batches(seed)
    w_unit = torch.randn(H); w_unit = (w_unit / w_unit.norm()).to(dev)
    base, scale = {}, {}
    with torch.no_grad():
        for li in LAYERS:
            base[li] = [layer_input(bi, li, batches) for bi in range(NB)]
            scale[li] = float(np.mean([base[li][bi].flatten(1).norm(dim=1).mean().item()
                                       for bi in range(NB)]))
    print(f"[seed {seed}] layer norms: " + " ".join(f"L{l}={scale[l]:.2f}" for l in LAYERS), flush=True)

    grid = [(a, L, li, m) for a in AMPS for L in LAGS for li in LAYERS for m in MODES]
    # L=0 正控
    grid += [(a, 0, li, m) for a in (0.004, 1.0) for li in LAYERS for m in ("natural", "force")]

    for a, L, li, mode in grid:
        blk = model.transformer[li]
        w = w_unit * (a * scale[li])
        outs, masses = [], []
        with torch.no_grad():
            for bi in range(NB):
                ho, ass = head_out(base[li][bi], blk.self_attn, blk.norm1, mode, L, w,
                                   batches[bi][3])
                outs.append((ho.numpy(), batches[bi][3].numpy()))
                masses.append(ass)
        best, besth = 0.0, -1
        for h in range(NH):
            Xtr, ytr, Xte, yte = [], [], [], []
            for bi, (ho, c) in enumerate(outs):
                feats = ho[:, h, L:, :].reshape(-1, DH)
                labels = c[:, :T - L].reshape(-1)
                if bi < 2: Xtr.append(feats); ytr.append(labels)
                else: Xte.append(feats); yte.append(labels)
            acc = ridge_probe(np.concatenate(Xtr), np.concatenate(ytr),
                              np.concatenate(Xte), np.concatenate(yte))
            if acc > best: best, besth = acc, h
        rows.append({"amp": a, "L": L, "layer": li, "mode": mode, "seed": seed,
                     "best_head_acc": round(best, 3), "head": besth,
                     "inj_ratio": round(float((w.norm() / scale[li]).item()), 5),
                     "attn_mass_on_target": (None if math.isnan(float(np.nanmean(masses)))
                                             else round(float(np.nanmean(masses)), 3))})
        print(f"  a={a:<6} L={L:<3} layer={li} {mode:<8} acc={best:.3f}", flush=True)

r = pd.DataFrame(rows)
r.to_csv(str(OUT).replace(".json", ".csv"), index=False)
json.dump({"meta": {"script": "m12_e3.py", "T": T, "B": B, "NB": NB, "seeds": SEEDS,
                    "amps": AMPS, "lags": LAGS, "layers": LAYERS, "beta": BETA,
                    "train_batches": 2, "test_batches": 2, "probe": "per-head ridge lam=10",
                    "runtime_s": round(time.time() - t0, 1)},
           "rows": rows}, open(OUT, "w"), indent=1)

print("\n=== mean best-head acc over 3 seeds ===")
p = r.pivot_table(index=["amp", "L", "layer"], columns="mode", values="best_head_acc", aggfunc="mean")
print(p.round(3).to_string())
print("\n=== L=0 positive control ===")
print(r[r["L"] == 0].pivot_table(index=["amp", "layer"], columns="mode",
                                 values="best_head_acc", aggfunc="mean").round(3).to_string())
print(f"\nsaved {OUT}  ({time.time()-t0:.0f}s)")