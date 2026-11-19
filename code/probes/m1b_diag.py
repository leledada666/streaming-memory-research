# 诊断（不改变研究设计）：阳性对照。检查探针管线能否读出 c_t（L=0），
# 以及注入在 attention 输出里的可见强度。只用于判断"全 0.50"是真实无信号还是实现问题。
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
print("d_model", H, "layers", NL, "heads", NH, "head_dim", DH)

batches = []
for bi in range(NB):
    st = rng.integers(int(N * 0.85), N - T, B)
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
    return att @ v, att


def ridge_probe(Xtr, ytr, Xte, yte, lam=10.0):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    A = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]
    Bm = np.c_[np.ones(len(Xte)), (Xte - mu) / sd]
    M = A.T @ A + lam * np.eye(A.shape[1]); M[0, 0] -= lam
    beta = np.linalg.solve(M, A.T @ ytr)
    return float((np.sign(Bm @ beta) == yte).mean())


def probe_sites(get_feats, c_list, lag):
    accs = []
    for h in range(NH):
        Xtr, ytr, Xte, yte = [], [], [], []
        for bi in range(NB):
            feats = get_feats(bi, h)[:, lag:, :].reshape(-1, DH) if lag > 0 else get_feats(bi, h).reshape(-1, DH)
            labels = c_list[bi][:, :T - lag].reshape(-1) if lag > 0 else c_list[bi].reshape(-1)
            if bi < 4: Xtr.append(feats); ytr.append(labels)
            else: Xte.append(feats); yte.append(labels)
        accs.append(ridge_probe(np.concatenate(Xtr), np.concatenate(ytr), np.concatenate(Xte), np.concatenate(yte)))
    return accs


for li in [0, 3, 7]:
    attn = model.transformer[li].self_attn
    ho_inj, ho_clean, xinj = [], [], []
    for bi, (s1, s2, stamp, c) in enumerate(batches):
        with torch.no_grad():
            x0 = model.embedding([s1, s2]) + model.time_emb(stamp)
            for j in range(li):
                x0 = model.transformer[j](x0)
            x1 = x0 + c[:, :, None] * w[None, None, :]
            a, _ = head_outputs(x1, attn)
            b_, _ = head_outputs(x0, attn)
        ho_inj.append(a.numpy()); ho_clean.append(b_.numpy()); xinj.append(x1.numpy())
        if bi == 0:
            d = (a - b_)
            print(f"layer {li}: ||ho_inj-ho_clean||/||ho_clean|| = {float(d.norm()/b_.norm()):.4f}, "
                  f"per-dim std ho={float(a.std()):.4f}, diff std={float(d.std()):.4f}")
    cs = [c.numpy() for _, _, _, c in batches]
    acc0 = probe_sites(lambda bi, h: ho_inj[bi][:, h, :, :], cs, 0)
    acc1 = probe_sites(lambda bi, h: ho_inj[bi][:, h, :, :], cs, 1)
    accx0 = probe_sites(lambda bi, h: xinj[bi], cs, 0)
    accx1 = probe_sites(lambda bi, h: xinj[bi], cs, 1)
    print(f"layer {li} L=0 on ho  : max={max(acc0):.3f} mean={np.mean(acc0):.3f} n>0.6={sum(a>0.6 for a in acc0)}")
    print(f"layer {li} L=1 on ho  : max={max(acc1):.3f} mean={np.mean(acc1):.3f} n>0.6={sum(a>0.6 for a in acc1)}")
    print(f"layer {li} L=0 on x_inj: max={max(accx0):.3f} mean={np.mean(accx0):.3f}")
    print(f"layer {li} L=1 on x_inj: max={max(accx1):.3f} mean={np.mean(accx1):.3f}")