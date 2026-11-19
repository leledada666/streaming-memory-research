# M12c: 跨层存储检查（排除"同层 force=1.0 恒真"）
# 在 embedding 后注入 c*w（层 0 尺度），穿过 block0..6（真实注意力+FFN），
# 在层 7 用 force/wrong/natural 读出。若 force 仍高 => 信息真的能存过 7 层。
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
READ, LAGS = 7, [0, 1, 10, 15, 30, 60]
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

def embed(bi):
    s1, s2, stamp, c = batches[bi]
    with torch.no_grad():
        return model.embedding([s1, s2]) + model.time_emb(stamp), c

SCALE0 = np.mean([embed(bi)[0].flatten(1).norm(dim=1).mean().item() for bi in range(NB)])
w0 = w_unit * SCALE0
print(f"inject scale (layer0 space): ||c*w||/||x0|| = 1.0 ({SCALE0:.1f})")

def head_outputs(x, blk, mode, L):
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
    mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
    return logits.masked_fill(~mask, float("-inf")).softmax(-1) @ v

def ridge_probe(Xtr, ytr, Xte, yte):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    A = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]; Bm = np.c_[np.ones(len(Xte)), (Xte - mu) / sd]
    lam = 10.0; M = A.T @ A + lam * np.eye(A.shape[1]); M[0, 0] -= lam
    beta = np.linalg.solve(M, A.T @ ytr)
    return float((np.sign(Bm @ beta) == yte).mean())

blk = model.transformer[READ]
rows = []
for L in LAGS:
    for mode in ["natural", "force", "wrong"]:
        outs = []
        for bi in range(NB):
            x0, c = embed(bi)
            with torch.no_grad():
                x = x0 + c[:, :, None] * w0[None, None, :]     # 只在这一处注入
                for j in range(READ):
                    x = model.transformer[j](x)              # 真实 7 层
                ho = head_outputs(x, blk, mode, L)
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
        rows.append({"L": L, "mode": mode, "acc": round(best, 3)})

r = pd.DataFrame(rows)
print("\n=== 跨层存储：注入@层0，读出@层7（best head）===")
print(r.pivot_table(index="L", columns="mode", values="acc").to_string())
r.to_csv(r"C:\Users\lele\AppData\Local\Temp\opencode\m12_cross.csv", index=False)