# 敏感性扫描（诊断，非原设计）：把注入幅度缩放到该层激活 std，全部 8 层 × 全部 L。
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
T = 96; B = 12; NB = 8; LAGS = [0, 1, 5, 10, 15, 30, 60]
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


rows = []
for li in range(NL):
    attn = model.transformer[li].self_attn
    base = []
    for bi, (s1, s2, stamp, c) in enumerate(batches):
        with torch.no_grad():
            x0 = model.embedding([s1, s2]) + model.time_emb(stamp)
            for j in range(li):
                x0 = model.transformer[j](x0)
        base.append(x0.numpy())
    sc = float(np.concatenate([b.reshape(-1, H) for b in base]).std())
    ho = []
    for bi in range(NB):
        with torch.no_grad():
            x = torch.tensor(base[bi]) + batches[bi][3][:, :, None] * (w * sc)[None, None, :]
            a, _ = head_outputs(x, attn)
        ho.append(a.numpy())
    for L in LAGS:
        accs = []
        for h in range(NH):
            Xtr, ytr, Xte, yte = [], [], [], []
            for bi in range(NB):
                feats = ho[bi][:, h, L:, :].reshape(-1, DH)
                labels = cs[bi][:, :T - L].reshape(-1)
                if bi < 4: Xtr.append(feats); ytr.append(labels)
                else: Xte.append(feats); yte.append(labels)
            accs.append(ridge_probe(np.concatenate(Xtr), np.concatenate(ytr), np.concatenate(Xte), np.concatenate(yte)))
        rows.append({"layer": li, "L": L, "x_std": round(sc, 2), "max_acc": round(max(accs), 3),
                     "mean_acc": round(float(np.mean(accs)), 3), "best_head": int(np.argmax(accs)),
                     "n_gt60": int(sum(a > 0.60 for a in accs)), "n_gt55": int(sum(a > 0.55 for a in accs))})
    print(f"layer {li} done (x_std={sc:.1f})", flush=True)

r = pd.DataFrame(rows)
r.to_csv(r"C:\Users\lele\AppData\Local\Temp\opencode\m1b_scaled_results.csv", index=False)
print("\n=== 最佳头 test acc (注入幅度=std(x)) ===")
print(r.pivot(index="layer", columns="L", values="max_acc").to_string())
print("\n=== 每层 >0.60 的头数 ===")
print(r.pivot(index="layer", columns="L", values="n_gt60").to_string())
print("\n=== 每层 >0.55 的头数 ===")
print(r.pivot(index="layer", columns="L", values="n_gt55").to_string())
agg = r.groupby("L").agg(max_acc=("max_acc", "max"), mean_max=("max_acc", "mean"),
                         mean_all=("mean_acc", "mean"), sites_gt60=("n_gt60", "sum"), sites_gt55=("n_gt55", "sum"))
print("\n=== 全部站点按 L 聚合 ===")
print(agg.to_string())