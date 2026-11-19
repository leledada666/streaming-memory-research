# M1b: 站点级读出探针（零训练，CPU）
# 给冻结 Kronos 的某一层输入注入已知 bit c_t * w（w 随机单位向量），
# 在每层每头的 attention 输出上做线性探针，测该站点能否读出 c_{t-L}。
# 目的：验证 L=10->15 的塌陷是否来自 "有信号的站点数 K_eff" 骤降。
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
T = 96; B = 12; NB = 8   # 8 个 batch，前 4 训练探针，后 4 测试
LAGS = [1, 5, 10, 15, 30, 60]
model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
H = model.d_model; NL = len(model.transformer); NH = model.transformer[0].self_attn.n_heads; DH = model.transformer[0].self_attn.head_dim
w = torch.randn(H); w = (w / w.norm()).to(dev)

# 预生成 batch：同一批 c 用于所有层（层间可比），但注入只在目标层
batches = []
for bi in range(NB):
    st = rng.integers(int(N*0.85), N-T, B)
    idx = torch.tensor(st[:, None] + np.arange(T))
    s1 = torch.tensor(z["S1"][idx.numpy()]).to(dev); s2 = torch.tensor(z["S2"][idx.numpy()]).to(dev)
    stamp = torch.tensor(STAMP[idx.numpy()]).to(dev)
    c = torch.tensor(rng.choice([-1.0, 1.0], size=(B, T)).astype(np.float32)).to(dev)
    batches.append((s1, s2, stamp, c))

def head_outputs(x, attn):
    b, t, _ = x.shape
    q = attn.q_proj(x).view(b, t, NH, DH).transpose(1, 2)
    k = attn.k_proj(x).view(b, t, NH, DH).transpose(1, 2)
    v = attn.v_proj(x).view(b, t, NH, DH).transpose(1, 2)
    q, k = attn.rotary(q, k, position_offset=0)
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(DH)
    mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
    att = logits.masked_fill(~mask, float("-inf")).softmax(-1)
    return att @ v, att   # B,H,T,DH ; B,H,T,T

def ridge_probe(Xtr, ytr, Xte, yte):
    # X: n,d ; y: n (±1)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    A = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]
    Bm = np.c_[np.ones(len(Xte)), (Xte - mu) / sd]
    lam = 10.0
    M = A.T @ A + lam * np.eye(A.shape[1]); M[0, 0] -= lam
    beta = np.linalg.solve(M, A.T @ ytr)
    pred = Bm @ beta
    return float((np.sign(pred) == yte).mean())

results = []
for li in range(NL):
    attn = model.transformer[li].self_attn
    # 收集该层所有 batch 的 head 输出；注入只在该层输入
    outs = []
    for bi, (s1, s2, stamp, c) in enumerate(batches):
        with torch.no_grad():
            x = model.embedding([s1, s2]) + model.time_emb(stamp)
            for j in range(li):
                x = model.transformer[j](x)
            x = x + c[:, :, None] * w[None, None, :]   # 注入已知 bit
            ho, att = head_outputs(x, attn)
        outs.append((ho.numpy(), c.numpy(), att.numpy()))
    for L in LAGS:
        if L >= T: continue
        accs = []; alphas = []
        for h in range(NH):
            # 训练探针：batch 0-3；测试：4-7；特征 t>=L，标签 c[:,t-L]
            Xtr, ytr, Xte, yte = [], [], [], []
            for bi, (ho, c, att) in enumerate(outs):
                feats = ho[:, h, L:, :].reshape(-1, DH)          # (B*(T-L), DH)
                labels = c[:, :T-L].reshape(-1)
                if bi < 4: Xtr.append(feats); ytr.append(labels)
                else: Xte.append(feats); yte.append(labels)
            Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
            Xte = np.concatenate(Xte); yte = np.concatenate(yte)
            acc = ridge_probe(Xtr, ytr, Xte, yte)
            alpha = float(np.mean([np.diagonal(a[:, h, L:, :T-L], axis1=-2, axis2=-1).mean() for a, _, _ in outs]))
            accs.append(acc); alphas.append(alpha)
        results.append({"layer": li, "L": L, "max_acc": round(max(accs), 3), "mean_acc": round(float(np.mean(accs)), 3),
                        "n_sites_gt55": int(sum(a > 0.55 for a in accs)), "n_sites_gt60": int(sum(a > 0.60 for a in accs)),
                        "best_head": int(np.argmax(accs)), "alpha_best": round(alphas[int(np.argmax(accs))], 5)})

r = pd.DataFrame(results)
print("\n=== 每个 (layer, L) 的最佳头准确率 ===")
print(r.pivot(index="layer", columns="L", values="max_acc").to_string())
print("\n=== 每层有多少头 test acc > 0.60（K_eff 量级）===")
print(r.pivot(index="layer", columns="L", values="n_sites_gt60").to_string())
print("\n=== 每层有多少头 > 0.55 ===")
print(r.pivot(index="layer", columns="L", values="n_sites_gt55").to_string())
print("\n=== 全部站点（层×头）中：最大值 / 均值 / >0.60 的站点数 ===")
agg = r.groupby("L").agg(max_acc=("max_acc", "max"), mean_acc=("mean_acc", "mean"),
                         sites_gt55=("n_sites_gt55", "sum"), sites_gt60=("n_sites_gt60", "sum")).reset_index()
print(agg.to_string(index=False))