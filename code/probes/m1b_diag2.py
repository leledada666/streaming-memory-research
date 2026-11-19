# 诊断 v2（不改变研究设计）：阳性对照 + 注入强度敏感性检查。
import math, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(r"E:\rwkv7"); sys.path.insert(0, str(ROOT / "kronos_src"))
from model import Kronos
dev = "cpu"; torch.set_num_threads(8)
torch.manual_seed(0); rng = np.random.default_rng(0)

df = pd.read_csv(ROOT / "data" / "binance" / "klines_1m" / "bars_1m_of_3m.csv", usecols=["ts"])
N = len(df)
z = np.load(ROOT / "runs" / "e6_long_2026-10-20" / "e5_tokens.npz")
ts = pd.to_datetime(df["ts"], unit="s", utc=True)
STAMP = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1).astype(np.float32)
T = 96; B = 12; NB = 8
model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
H = model.d_model; NL = len(model.transformer)
NH = model.transformer[0].self_attn.n_heads; DH = model.transformer[0].self_attn.head_dim
w = torch.randn(H); w = (w / w.norm()).to(dev)

batches = []
for bi in range(NB):
    st = rng.integers(int(N * 0.85), N - T, B)
    idx = torch.tensor(st[:, None] + np.arange(T))
    s1 = torch.tensor(z["S1"][idx.numpy()]).to(dev); s2 = torch.tensor(z["S2"][idx.numpy()]).to(dev)
    stamp = torch.tensor(STAMP[idx.numpy()]).to(dev)
    c = torch.tensor(rng.choice([-1.0, 1.0], size=(B, T)).astype(np.float32)).to(dev)
    batches.append((s1, s2, stamp, c))
cs = [c.numpy() for _, _, _, c in batches]


def head_outputs(x, attn):
    b, t, _ = x.shape
    q = attn.q_proj(x).view(b, t, NH, DH).transpose(1, 2)
    k = attn.k_proj(x).view(b, t, NH, DH).transpose(1, 2)
    v = attn.v_proj(x).view(b, t, NH, DH).transpose(1, 2)
    q, k = attn.rotary(q, k, position_offset=0)
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(DH)
    mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
    att = logits.masked_fill(~mask, float("-inf")).softmax(-1)
    return att @ v, att


def ridge_probe(Xtr, ytr, Xte, yte, lam=10.0):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    A = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]
    Bm = np.c_[np.ones(len(Xte)), (Xte - mu) / sd]
    M = A.T @ A + lam * np.eye(A.shape[1]); M[0, 0] -= lam
    beta = np.linalg.solve(M, A.T @ ytr)
    return float((np.sign(Bm @ beta) == yte).mean())


def full_probe(feat_list, lag):
    Xtr, ytr, Xte, yte = [], [], [], []
    for bi in range(NB):
        f = feat_list[bi]
        feats = f[:, lag:, :].reshape(-1, f.shape[-1])
        labels = cs[bi][:, :T - lag].reshape(-1)
        if bi < 4: Xtr.append(feats); ytr.append(labels)
        else: Xte.append(feats); yte.append(labels)
    return ridge_probe(np.concatenate(Xtr), np.concatenate(ytr), np.concatenate(Xte), np.concatenate(yte))


for li in [0, 3, 7]:
    attn = model.transformer[li].self_attn
    base = []
    for bi, (s1, s2, stamp, c) in enumerate(batches):
        with torch.no_grad():
            x0 = model.embedding([s1, s2]) + model.time_emb(stamp)
            for j in range(li):
                x0 = model.transformer[j](x0)
        base.append(x0.numpy())
    sc = float(np.concatenate([b.reshape(-1, H) for b in base]).std())
    print(f"\n--- layer {li}: x per-dim std = {sc:.3f} ---")

    xinj = [base[bi] + cs[bi][:, :, None] * (w.numpy() * sc)[None, None, :] for bi in range(NB)]
    print(f"layer {li}: 阳性对照 probe(x_inj, 尺度={sc:.2f}) L=0 acc = {full_probe(xinj, 0):.3f} (期望≈1.0)")
    print(f"layer {li}: probe(x_inj) L=1 acc = {full_probe(xinj, 1):.3f} (x_t 不含 c_t-1，期望≈0.5)")

    for name, scale in [("w 单位范数(原设计)", 1.0), (f"w*std(x)", sc)]:
        ho = []
        for bi in range(NB):
            with torch.no_grad():
                x = torch.tensor(base[bi]) + batches[bi][3][:, :, None] * (w * scale)[None, None, :]
                a, _ = head_outputs(x, attn)
            ho.append(a.numpy())
        cells = []
        for L in [0, 1, 5, 10, 15, 30, 60]:
            accs = [full_probe([ho[bi][:, h, :, :] for bi in range(NB)], L) for h in range(NH)]
            cells.append(f"L={L}:{max(accs):.3f}/{np.mean(accs):.3f}/{sum(a>0.6 for a in accs)}")
        print(f"layer {li} ho [{name}] (max/mean/n>0.60) " + " ".join(cells))