# -*- coding: utf-8 -*-
"""FIR-16 延迟线对照 (CPU): 用最近 16 个 m_t=[c_t,z_t] 预测 c_{t-L}
若 L=15 的 FIR-16(33 参数)就达到 ~1.0, 则 GRU 的 .981 只证明"15-tap 延迟线", 不证明长程记忆。
数据: 与 E6 相同切分 (train [0,.70N), val [.70N,.85N), test [.85N,N)), c iid +-1, z~N(0,1)。
臂:
  A  FIR-16 固定 lag, 32 维闭式岭回归 (33 参数), L=0/5/10/15/30
  B  FIR-16 + lag cue (onehot x taps, 528 维, 513 参数), 训练 {5,10} -> 测 5/10/15/30
  C  参数对齐 FIR-16-MLP (32->2048->1, 69,633 参数 = GRU-S16 的 0.99x), L=15/30
  D  单 tap oracle (只取第 L 个 tap) = 该任务的上界
落盘: fir16.json
"""
import json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

OUT = Path(r"C:\Users\lele\AppData\Local\Temp\opencode\fir16.json")
torch.set_num_threads(8)
N = 132480
I1, I2 = int(N * 0.70), int(N * 0.85)
T, W = 32, 16
LAGS = [0, 5, 10, 15, 30]
NTR, NVA, NTE = 8000, 3000, 12000
GRU_S16_PARAMS = 70353          # e7_gru_S16_1bit_T32_s0.json
GRU_ACC15 = 0.981               # 上游报告值


def draw(n, lo, hi, seed):
    g = np.random.default_rng(seed)
    s = g.integers(lo, hi - T - W, n)
    c = g.choice([-1.0, 1.0], size=(n, T + W)).astype(np.float32)
    z = g.standard_normal((n, T + W)).astype(np.float32)
    return np.stack([c, z], -1), c


def taps(m):
    """[n, T+16, 2] -> [n, T, 32] (每个 t 取 j-15..j, j=16+t, 时间 x 通道)"""
    v = np.lib.stride_tricks.sliding_window_view(m, W, axis=1)   # [n, T+1, 2, W]
    v = v.transpose(0, 1, 3, 2).reshape(m.shape[0], T + 1, W * 2)
    return v[:, 1:T + 1, :]


def labels(c, L):
    n = c.shape[0]
    base = 16 + np.arange(T)
    return c[np.arange(n)[:, None], (base - L)[None, :]]


def ridge(Xtr, ytr, Xte, yte, lam):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    A = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]
    Bm = np.c_[np.ones(len(Xte)), (Xte - mu) / sd]
    M = A.T @ A + lam * np.eye(A.shape[1]); M[0, 0] -= lam
    beta = np.linalg.solve(M, A.T @ ytr)
    return float((np.sign(Bm @ beta) == yte).mean())


t0 = time.time()
mtr, ctr = draw(NTR, 0, I1, 11)
mva, cva = draw(NVA, I1, I2, 12)
mte, cte = draw(NTE, I2, N, 13)
Xtr, Xva, Xte = taps(mtr), taps(mva), taps(mte)
Xtr2, Xva2 = Xtr.reshape(-1, W * 2), Xva.reshape(-1, W * 2)
Xte2 = Xte.reshape(-1, W * 2)
print(f"data {Xtr2.shape} {Xva2.shape} {Xte2.shape}  ({time.time()-t0:.1f}s)", flush=True)

rows = []


def add(arm, L, acc, params, note=""):
    rows.append({"arm": arm, "L": L, "test_acc": round(acc, 4), "params": params, "note": note})
    print(f"{arm:<22} L={L:<3} test_acc={acc:.4f} params={params}  {note}", flush=True)


# ---- A: 固定 lag 的 32 维 ridge (33 参数) ----
for L in LAGS:
    ytr, yva, yte = labels(ctr, L).reshape(-1), labels(cva, L).reshape(-1), labels(cte, L).reshape(-1)
    acc_va = ridge(Xtr2, ytr, Xva2, yva, 10.0)
    add("A_fir16_ridge", L, ridge(Xtr2, ytr, Xte2, yte, 10.0), W * 2 + 1, f"val={acc_va:.3f}")

# ---- D: 单 tap oracle ----
for L in LAGS:
    if L >= W: continue
    tap = Xte[:, :, 2 * (W - 1 - L)].reshape(-1)
    add("D_tap_only_oracle", L, float((np.sign(tap) == labels(cte, L).reshape(-1)).mean()), 1, "上界")

# ---- B: lag-cue 块对角 (528 维, 513 参数), 训练 {5,10} ----
def cue_block(X, L):
    """[n,T,32] -> [n,T,16*32+16] 只填 cue=L 的块; 末尾追加 onehot"""
    n, t, d = X.shape
    out = np.zeros((n, t, 16 * (d + 1)), np.float32)
    out[..., L * d:(L + 1) * d] = X
    out[..., -16 + L] = 1.0
    return out.reshape(-1, 16 * (d + 1))


nb = 1200
Btr = np.concatenate([cue_block(Xtr[:nb], 5), cue_block(Xtr[:nb], 10)])
Bva = np.concatenate([cue_block(Xva[:400], 5), cue_block(Xva[:400], 10)])
ytr = np.concatenate([labels(ctr[:nb], 5).reshape(-1), labels(ctr[:nb], 10).reshape(-1)])
yva = np.concatenate([labels(cva[:400], 5).reshape(-1), labels(cva[:400], 10).reshape(-1)])
for L in (5, 10, 15):
    acc_va = ridge(Btr, ytr, Bva, yva, 10.0)
    add("B_fir16_cue_ridge", L,
        ridge(Btr, ytr, cue_block(Xte[:2000], L), labels(cte[:2000], L).reshape(-1), 10.0),
        Btr.shape[1], f"train_lags=5,10 (unseen L -> 0 块) val={acc_va:.3f}")

# ---- C: 参数对齐 MLP (32 -> 2048 -> 1) ----
class MLP(nn.Module):
    def __init__(self, P=2048):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(W * 2, P), nn.ReLU(), nn.Linear(P, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


mm, cm = draw(30000, 0, I1, 21)
Xm = torch.tensor(taps(mm).reshape(-1, W * 2))
params_c = W * 2 * 2048 + 2048 + 2048 + 1
for L in (15, 30):
    ym = torch.tensor((labels(cm, L).reshape(-1) > 0).astype(np.float32))
    torch.manual_seed(0)
    net = MLP(); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(1)
    for step in range(300):
        j = torch.randint(0, len(Xm), (4096,), generator=g)
        loss = lossf(net(Xm[j]), ym[j])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0: print(f"  C L={L} step {step} loss={loss.item():.4f}", flush=True)
    yte01 = torch.tensor((labels(cte, L).reshape(-1) > 0).astype(np.float32))
    correct = 0
    with torch.no_grad():
        for k in range(0, len(Xte2), 4096):
            lg = net(torch.tensor(Xte2[k:k + 4096]))
            correct += int(((lg > 0).float() == yte01[k:k + 4096]).sum())
    add("C_fir16_mlp_matched", L, correct / len(Xte2), params_c,
        f"param_ratio_vs_GRU={params_c/GRU_S16_PARAMS:.2f}")

json.dump({"meta": {"script": "fir16.py", "T": T, "W": W, "splits": [0, I1, I2, N],
                    "windows_tr/va/te": [NTR, NVA, NTE],
                    "gru_ref": {"params": GRU_S16_PARAMS, "acc_L15": GRU_ACC15},
                    "runtime_s": round(time.time() - t0, 1)},
           "rows": rows}, open(OUT, "w"), indent=1)
print(f"\nsaved {OUT} ({time.time()-t0:.0f}s)")