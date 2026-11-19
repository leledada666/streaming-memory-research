# M12b: storage vs addressing 因果探针（零训练，CPU），修正 M12 的幅度问题
# 修正点（相对 m12_storage_vs_address.py）：
#  1) 注入幅度按层缩放：w_l = unit_w * mean_t||x_t||（每维 std 同量级），并打印 ||c*w||/||x||
#  2) head_outputs 走真实块路径：先 attn.norm1(x) 再 q/k/v（原来漏了 pre-attn RMSNorm，
#     导致 logits 尺度失真、+beta 不可靠）
#  3) 加 L=0 正控（目标在当前位置），force 在 L=0 必须 >0.9，否则说明管线/信号仍有问题
#  4) wrong 的错 lag：max(1, L//2)，L=1 时用 2，避免与 force 重合
#  5) BETA=40，并打印 force 在目标 key 上的注意力质量，确认"强制寻址"真的发生
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
LAGS = [0, 1, 10, 15, 30, 60]
LAYERS = [0, 3, 7]
BETA = 40.0
model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
H = model.d_model; NL = len(model.transformer); NH = model.transformer[0].self_attn.n_heads; DH = model.transformer[0].self_attn.head_dim
w_unit = torch.randn(H); w_unit = (w_unit / w_unit.norm()).to(dev)

batches = []
for bi in range(NB):
    st = rng.integers(int(N * 0.85), N - T, B)
    idx = torch.tensor(st[:, None] + np.arange(T))
    s1 = torch.tensor(z["S1"][idx.numpy()]).to(dev); s2 = torch.tensor(z["S2"][idx.numpy()]).to(dev)
    stamp = torch.tensor(STAMP[idx.numpy()]).to(dev)
    c = torch.tensor(rng.choice([-1.0, 1.0], size=(B, T)).astype(np.float32)).to(dev)
    batches.append((s1, s2, stamp, c))

def layer_input(bi, li):
    s1, s2, stamp, c = batches[bi]
    with torch.no_grad():
        x = model.embedding([s1, s2]) + model.time_emb(stamp)
        for j in range(li):
            x = model.transformer[j](x)
    return x

# ---- 1) 每层激活范数 & 注入缩放 ----
print("=== 层输入范数 (mean over B*T) ===")
SCALE = {}
for li in LAYERS:
    nrm, std = [], []
    for bi in range(NB):
        x = layer_input(bi, li)
        nrm.append(x.flatten(1).norm(dim=1)); std.append(x.reshape(-1, H).std(0))
    nrm = torch.cat(nrm).mean().item(); sd = torch.stack(std).mean(0)
    SCALE[li] = nrm
    print(f"layer {li}: mean||x_t||={nrm:.3f}  mean per-dim std={sd.mean().item():.4f}  "
          f"max dim std={sd.max().item():.4f}  <-- w 缩放到 {nrm:.3f}")

def head_outputs(x, attn, norm1, mode, L, w=None, c=None, stats=None):
    if w is not None:
        x = x + c[:, :, None] * w[None, None, :]
    xn = norm1(x)                           # 真实块路径：pre-attn RMSNorm
    b, t, _ = xn.shape
    q = attn.q_proj(xn).view(b, t, NH, DH).transpose(1, 2)
    k = attn.k_proj(xn).view(b, t, NH, DH).transpose(1, 2)
    v = attn.v_proj(xn).view(b, t, NH, DH).transpose(1, 2)
    q, k = attn.rotary(q, k, position_offset=0)
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(DH)
    lag = None
    if mode in ("force", "wrong"):
        lag = L if mode == "force" else (L // 2 if L // 2 >= 1 else 2)
        for tq in range(lag, t):
            logits[:, :, tq, tq - lag] += BETA
    mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
    att = logits.masked_fill(~mask, float("-inf")).softmax(-1)
    if stats is not None and lag is not None and mode == "force":
        rows = torch.arange(lag, t)
        stats.append(att[:, :, rows, rows - lag].mean().item())
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
    blk = model.transformer[li]; attn = blk.self_attn; norm1 = blk.norm1
    w = w_unit * SCALE[li]
    x0 = layer_input(0, li)
    print(f"\nlayer {li}: 注入范数比 ||c*w||/||x|| = {(w.norm()/x0.flatten(1).norm(dim=1).mean()):.3f}")
    for L in LAGS:
        for mode in ["natural", "force", "wrong"]:
            outs = []; mass = []
            for bi in range(NB):
                s1, s2, stamp, c = batches[bi]
                with torch.no_grad():
                    x = layer_input(bi, li)
                    # 三种 mode 都注入 c*w；mode 只改 attention logits
                    ho, att = head_outputs(x, attn, norm1, mode, L, w=w, c=c, stats=mass)
                outs.append((ho.numpy(), c.numpy()))
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
            m = np.mean(mass) if mass else float("nan")
            rows.append({"layer": li, "L": L, "mode": mode, "best_head_acc": round(best, 3),
                         "head": besth, "attn_mass_on_target": round(m, 3)})

r = pd.DataFrame(rows)
print("\n=== best-head probe acc：natural / force / wrong ===")
print(r.pivot_table(index=["layer", "L"], columns="mode", values="best_head_acc").to_string())
print("\n=== force 在目标 key 上的注意力质量（应≈1）===")
print(r[r["mode"] == "force"].pivot_table(index="L", columns="layer",
                                           values="attn_mass_on_target").to_string())
print("\n=== 跨层取最大 ===")
print(r.groupby(["L", "mode"]).best_head_acc.max().unstack().to_string())
r.to_csv(r"C:\Users\lele\AppData\Local\Temp\opencode\m12_scaled.csv", index=False)