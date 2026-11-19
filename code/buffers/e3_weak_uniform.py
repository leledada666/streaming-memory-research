# E3: 弱写 + uniform 对照（CPU）
# a=0.004 约等于自然写入量级；若 force 在此仍高，则"存储不是瓶颈"可锁死；
# 若只有 a=1.0 才高，则 1.000 只是上界，写入增益是隐藏超参。
import math, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(r"E:\rwkv7"); sys.path.insert(0, str(ROOT / "kronos_src"))
from model import Kronos
dev = "cpu"; torch.set_num_threads(8)
torch.manual_seed(0); rng = np.random.default_rng(0)

df = pd.read_csv(ROOT / "data" / "binance" / "klines_1m" / "bars_1m_of_3m.csv", usecols=["ts"])
N = len(df); z = np.load(ROOT / "runs" / "e6_long_2026-10-20" / "e5_tokens.npz")
ts = pd.to_datetime(df["ts"], unit="s", utc=True)
STAMP = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1).astype(np.float32)
T, B, NB, BETA = 96, 8, 4, 40.0
ALPHAS = [0.004, 0.01, 0.02, 1.0]
LAGS = [0, 15, 60]
LAYERS = [0, 7]
MODES = ["natural", "force", "wrong", "uniform"]
model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
H = model.d_model; NH = model.transformer[0].self_attn.n_heads; DH = model.transformer[0].self_attn.head_dim
w_unit = torch.randn(H); w_unit = (w_unit / w_unit.norm()).to(dev)

batches = []
for bi in range(NB):
    st = rng.integers(int(N * 0.85), N - T, B)
    idx = torch.tensor(st[:, None] + np.arange(T))
    batches.append((torch.tensor(z["S1"][idx.numpy()]).to(dev), torch.tensor(z["S2"][idx.numpy()]).to(dev),
                    torch.tensor(STAMP[idx.numpy()]).to(dev),
                    torch.tensor(rng.choice([-1.0, 1.0], size=(B, T)).astype(np.float32)).to(dev)))

def layer_input(bi, li):
    s1, s2, stamp, c = batches[bi]
    with torch.no_grad():
        x = model.embedding([s1, s2]) + model.time_emb(stamp)
        for j in range(li):
            x = model.transformer[j](x)
    return x, c

STATS = {}
for li in LAYERS:
    nrm, sd = [], []
    for bi in range(NB):
        x, _ = layer_input(bi, li)
        nrm.append(x.flatten(1).norm(dim=1)); sd.append(x.reshape(-1, H).std(0))
    STATS[li] = (torch.cat(nrm).mean().item(), torch.stack(sd).mean(0).mean().item())
    print(f"layer {li}: mean||x||={STATS[li][0]:.1f} per-dim std={STATS[li][1]:.4f}", flush=True)

def head_outputs(x, blk, mode, L, w, c):
    x = x + c[:, :, None] * w[None, None, :]
    xn = blk.norm1(x); attn = blk.self_attn
    b, t, _ = xn.shape
    q = attn.q_proj(xn).view(b, t, NH, DH).transpose(1, 2)
    k = attn.k_proj(xn).view(b, t, NH, DH).transpose(1, 2)
    v = attn.v_proj(xn).view(b, t, NH, DH).transpose(1, 2)
    q, k = attn.rotary(q, k, position_offset=0)
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(DH)
    if mode in ("force", "wrong"):
        lag = L if mode == "force" else (L // 2 if L // 2 >= 1 else 2)
        for tq in range(lag, t):
            logits[:, :, tq, tq - lag] += BETA
    elif mode == "uniform":
        logits = torch.zeros_like(logits)
    mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
    att = logits.masked_fill(~mask, float("-inf")).softmax(-1)
    return att @ v

def ridge_probe(Xtr, ytr, Xte, yte):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    A = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]; Bm = np.c_[np.ones(len(Xte)), (Xte - mu) / sd]
    lam = 10.0; M = A.T @ A + lam * np.eye(A.shape[1]); M[0, 0] -= lam
    beta = np.linalg.solve(M, A.T @ ytr)
    return float((np.sign(Bm @ beta) == yte).mean())

rows = []
for li in LAYERS:
    blk = model.transformer[li]; nrm, sd = STATS[li]
    for a in ALPHAS:
        w = w_unit * (a * nrm)
        dim_ratio = (a * nrm / math.sqrt(H)) / sd
        print(f"\nlayer {li} a={a}: ||c*w||/||x||={a:.3f} per-dim sig/std={dim_ratio:.3f}", flush=True)
        for L in LAGS:
            for mode in MODES:
                outs = []
                for bi in range(NB):
                    x, c = layer_input(bi, li)
                    with torch.no_grad():
                        ho = head_outputs(x, blk, mode, L, w, c)
                    outs.append((ho.numpy(), c.numpy()))
                best = 0.0
                for h in range(NH):
                    Xtr, ytr, Xte, yte = [], [], [], []
                    for bi, (ho, c) in enumerate(outs):
                        feats = ho[:, h, L:, :].reshape(-1, DH); labels = c[:, :T - L].reshape(-1)
                        if bi < 2: Xtr.append(feats); ytr.append(labels)
                        else: Xte.append(feats); yte.append(labels)
                    best = max(best, ridge_probe(np.concatenate(Xtr), np.concatenate(ytr),
                                                 np.concatenate(Xte), np.concatenate(yte)))
                rows.append({"layer": li, "a": a, "L": L, "mode": mode,
                             "acc": round(best, 3), "dim_sig_std": round(dim_ratio, 2)})

r = pd.DataFrame(rows)
for li in LAYERS:
    print(f"\n-- layer {li} --")
    print(r[r.layer == li].pivot_table(index=["a", "L"], columns="mode", values="acc").to_string())
out = r"C:\Users\lele\AppData\Local\Temp\opencode\e3_weak_uniform.csv"
r.to_csv(out, index=False)
print("\nsaved", out)