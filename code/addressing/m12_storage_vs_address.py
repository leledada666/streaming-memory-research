# M12: storage vs addressing 的因果判决（零训练，CPU）
# 在冻结 Kronos 的第 l 层输入注入已知 bit c_s*w；对比三种读出：
#   natural : 原始冻结注意力
#   force   : 在 attention logits 上给 (q=t, k=t-L) 加 +beta，强制寻址到目标
#   wrong   : 加 +beta 到 (q=t, k=t-Lr)（错误 lag），排除"任意偏置都能提升"
# 在每个 (layer, head) 的 attention 输出上做线性探针读 c_{t-L}。
# 若 force >> natural 且 wrong≈natural => 信息存得下，瓶颈是寻址。
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
T = 96; B = 8; NB = 4
LAGS = [1, 10, 15, 30, 60]
LAYERS = [0, 3, 7]
BETA = 20.0
model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
H = model.d_model; NL = len(model.transformer); NH = model.transformer[0].self_attn.n_heads; DH = model.transformer[0].self_attn.head_dim
w = torch.randn(H); w = (w / w.norm()).to(dev)

batches = []
for bi in range(NB):
    st = rng.integers(int(N*0.85), N-T, B)
    idx = torch.tensor(st[:, None] + np.arange(T))
    s1 = torch.tensor(z["S1"][idx.numpy()]).to(dev); s2 = torch.tensor(z["S2"][idx.numpy()]).to(dev)
    stamp = torch.tensor(STAMP[idx.numpy()]).to(dev)
    c = torch.tensor(rng.choice([-1.0, 1.0], size=(B, T)).astype(np.float32)).to(dev)
    batches.append((s1, s2, stamp, c))

def head_outputs(x, attn, mode, L):
    b, t, _ = x.shape
    q = attn.q_proj(x).view(b, t, NH, DH).transpose(1, 2)
    k = attn.k_proj(x).view(b, t, NH, DH).transpose(1, 2)
    v = attn.v_proj(x).view(b, t, NH, DH).transpose(1, 2)
    q, k = attn.rotary(q, k, position_offset=0)
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(DH)
    if mode in ("force", "wrong"):
        lag = L if mode == "force" else max(1, L // 2)
        for tq in range(lag, t):
            logits[:, :, tq, tq - lag] += BETA
    mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
    att = logits.masked_fill(~mask, float("-inf")).softmax(-1)
    return att @ v, att

def ridge_probe(Xtr, ytr, Xte, yte):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    A = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]; Bm = np.c_[np.ones(len(Xte)), (Xte - mu) / sd]
    lam = 10.0
    M = A.T @ A + lam * np.eye(A.shape[1]); M[0, 0] -= lam
    beta = np.linalg.solve(M, A.T @ ytr)
    return float((np.sign(Bm @ beta) == yte).mean())

rows = []
for li in LAYERS:
    attn = model.transformer[li].self_attn
    for L in LAGS:
        for mode in ["natural", "force", "wrong"]:
            outs = []
            for bi, (s1, s2, stamp, c) in enumerate(batches):
                with torch.no_grad():
                    x = model.embedding([s1, s2]) + model.time_emb(stamp)
                    for j in range(li):
                        x = model.transformer[j](x)
                    x = x + c[:, :, None] * w[None, None, :]
                    ho, att = head_outputs(x, attn, mode, L)
                outs.append((ho.numpy(), c.numpy()))
            # 每个 head 分别做探针，取最大
            best = 0.0
            for h in range(NH):
                Xtr, ytr, Xte, yte = [], [], [], []
                for bi, (ho, c) in enumerate(outs):
                    feats = ho[:, h, L:, :].reshape(-1, DH)
                    labels = c[:, :T-L].reshape(-1)
                    if bi < 2: Xtr.append(feats); ytr.append(labels)
                    else: Xte.append(feats); yte.append(labels)
                acc = ridge_probe(np.concatenate(Xtr), np.concatenate(ytr),
                                  np.concatenate(Xte), np.concatenate(yte))
                best = max(best, acc)
            rows.append({"layer": li, "L": L, "mode": mode, "best_head_acc": round(best, 3)})

r = pd.DataFrame(rows)
print("\n=== best-head probe acc：natural / force / wrong ===")
print(r.pivot_table(index=["layer", "L"], columns="mode", values="best_head_acc").to_string())
print("\n=== 跨层取最大 ===")
print(r.groupby(["L", "mode"]).best_head_acc.max().unstack().to_string())