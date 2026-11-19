# m1b_clean.py —— 零训练、随机方向的"自然读"探针（幅度对齐 + L=0 正控 + 残差流站点）
# 站点：每层 8 个 head 的 attention 输出；每层 1 个残差流输出（TransformerBlock 之后）。
# 注入：x <- x + c_t * sc * w，sc = std(x_layer)（逐维），因此注入/激活比 = 1.0（要求 >= 0.5）。
# 3 个 seed（w 方向与 batch 都不同），前 4 batch 训练 ridge，后 4 batch 测试。
import math, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(r"E:\rwkv7"); sys.path.insert(0, str(ROOT / "kronos_src"))
from model import Kronos
dev = "cpu"; torch.set_num_threads(8)

df = pd.read_csv(ROOT / "data" / "binance" / "klines_1m" / "bars_1m_of_3m.csv", usecols=["ts"])
N = len(df)
z = np.load(ROOT / "runs" / "e6_long_2026-10-20" / "e5_tokens.npz")
ts = pd.to_datetime(df["ts"], unit="s", utc=True)
STAMP = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1).astype(np.float32)
T = 96; B = 12; NB = 8; LAGS = [0, 1, 5, 10, 15, 30, 60]
SEEDS = [0, 1, 2]
model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
H = model.d_model; NL = len(model.transformer)
NH = model.transformer[0].self_attn.n_heads; DH = model.transformer[0].self_attn.head_dim
print(f"d_model={H} layers={NL} heads={NH} head_dim={DH}", flush=True)


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


def probe(feat_list, cs, lag):
    """feat_list[bi] -> (B,T,D)；返回 test acc"""
    Xtr, ytr, Xte, yte = [], [], [], []
    for bi in range(NB):
        f = feat_list[bi]
        feats = f[:, lag:, :].reshape(-1, f.shape[-1])
        labels = cs[bi][:, :T - lag].reshape(-1)
        if bi < 4: Xtr.append(feats); ytr.append(labels)
        else: Xte.append(feats); yte.append(labels)
    return ridge_probe(np.concatenate(Xtr), np.concatenate(ytr), np.concatenate(Xte), np.concatenate(yte))


rows = []
gate_min = 9.0; ratio_min = 9.0
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(H, generator=g); w = (w / w.norm()).to(dev)
    batches = []
    for bi in range(NB):
        st = rng.integers(int(N * 0.85), N - T, B)
        idx = torch.tensor(st[:, None] + np.arange(T))
        s1 = torch.tensor(z["S1"][idx.numpy()]).to(dev); s2 = torch.tensor(z["S2"][idx.numpy()]).to(dev)
        stamp = torch.tensor(STAMP[idx.numpy()]).to(dev)
        c = torch.tensor(rng.choice([-1.0, 1.0], size=(B, T)).astype(np.float32)).to(dev)
        batches.append((s1, s2, stamp, c))
    cs = [c.numpy() for _, _, _, c in batches]

    for li in range(NL):
        attn = model.transformer[li].self_attn
        x_in = []
        for bi, (s1, s2, stamp, c) in enumerate(batches):
            with torch.no_grad():
                x = model.embedding([s1, s2]) + model.time_emb(stamp)
                for j in range(li):
                    x = model.transformer[j](x)
            x_in.append(x)  # tensor (B,T,H)
        sc = float(torch.cat([xx.reshape(-1, H) for xx in x_in]).std())
        # 幅度对齐：w 为单位向量（逐维 rms=1/sqrt(H)），乘 sc*sqrt(H) 后逐维 rms = sc
        amp = sc * math.sqrt(H)
        ratio = float((w * amp).std()) / sc   # 逐维 RMS 比（=1.0）
        ratio_min = min(ratio_min, ratio)
        x_inj, ho, x_out, a_tt = [], [], [], []
        for bi in range(NB):
            with torch.no_grad():
                xi = x_in[bi] + batches[bi][3][:, :, None] * (w * amp)[None, None, :]
                a, att = head_outputs(xi, attn)
                xo = model.transformer[li](xi)
            x_inj.append(xi.numpy()); ho.append(a.numpy()); x_out.append(xo.numpy())
            if bi == 0:
                t = T
                idx = torch.arange(1, t)
                a_tt.append(float(att[:, :, idx, idx].mean()))
        a_tt = float(np.mean(a_tt))
        # 相对范数比（逐样本）：||inj|| / ||x||
        rel = float(np.mean([float((batches[bi][3][:, :, None] * (w * amp)).norm(dim=-1).mean()
                                 / x_in[bi].norm(dim=-1).mean()) for bi in range(NB)]))
        # L=0 正控：注入后的层输入（管线/对齐检查）
        gate = probe(x_inj, cs, 0)
        gate_min = min(gate_min, gate)
        head_l0 = max(probe([ho[bi][:, h, :, :] for bi in range(NB)], cs, 0) for h in range(NH))
        res_l0 = probe(x_out, cs, 0)
        for L in LAGS:
            accs = [probe([ho[bi][:, h, :, :] for bi in range(NB)], cs, L) for h in range(NH)]
            rows.append({"seed": seed, "layer": li, "L": L, "site": "head", "max_acc": max(accs),
                         "mean_acc": float(np.mean(accs)), "n_gt60": int(sum(a > 0.60 for a in accs)),
                         "n_gt55": int(sum(a > 0.55 for a in accs)), "res_acc": res_l0 if L == 0 else np.nan,
                         "gate_l0": gate if L == 0 else np.nan, "ratio": ratio, "a_tt": a_tt})
            rows.append({"seed": seed, "layer": li, "L": L, "site": "residual", "max_acc": probe(x_out, cs, L),
                         "mean_acc": np.nan, "n_gt60": np.nan, "n_gt55": np.nan, "res_acc": np.nan,
                         "gate_l0": np.nan, "ratio": ratio, "a_tt": a_tt})
        print(f"seed {seed} layer {li}: x_std={sc:.1f} amp={amp:.1f} ratio_rms={ratio:.2f} rel={rel:.2f} "
              f"gate_L0={gate:.3f} headL0={head_l0:.3f} resL0={res_l0:.3f} mean_a_tt={a_tt:.3f}", flush=True)

r = pd.DataFrame(rows)
r.to_csv(r"C:\Users\lele\AppData\Local\Temp\opencode\m1b_clean_results.csv", index=False)
hd = r[r.site == "head"]; rs = r[r.site == "residual"]

print(f"\n注入/激活逐维 RMS 比 min={ratio_min:.3f} (要求 >=0.5)；L=0 正控(层输入) 最低={gate_min:.3f}")
assert ratio_min >= 0.5 and gate_min > 0.9, "注入比或正控未达标，需先查管线"

# 主表：跨 seed 的 mean±sd（head 站点取该 L 下所有 layer×head 的最大值）
print("\n=== 主表（3 seed mean±sd）===")
line = []
for L in LAGS:
    a = hd[hd.L == L]
    per_seed_max = a.groupby("seed")["max_acc"].max()
    per_seed_gt60 = a.groupby("seed")["n_gt60"].sum()
    per_seed_gt55 = a.groupby("seed")["n_gt55"].sum()
    per_seed_res = rs[rs.L == L].groupby("seed")["max_acc"].max()
    line.append({
        "L": L,
        "best_head": f"{per_seed_max.mean():.3f}±{per_seed_max.std(ddof=1):.3f}",
        "sites_gt60": f"{per_seed_gt60.mean():.1f}±{per_seed_gt60.std(ddof=1):.1f}",
        "sites_gt55": f"{per_seed_gt55.mean():.1f}±{per_seed_gt55.std(ddof=1):.1f}",
        "residual_best": f"{per_seed_res.mean():.3f}±{per_seed_res.std(ddof=1):.3f}",
    })
print(pd.DataFrame(line).to_string(index=False))

print("\n=== 每层 best-head（seed 平均）/ 残差流（seed 平均）===")
tab = hd.groupby(["layer", "L"])["max_acc"].mean().unstack()
print(tab.round(3).to_string())
print("\n--- 残差流 ---")
tab2 = rs.groupby(["layer", "L"])["max_acc"].mean().unstack()
print(tab2.round(3).to_string())
print("\n=== L 聚合：跨 seed 的站点分布 ===")
agg = hd.groupby("L").agg(mean_of_mean=("mean_acc", "mean"), max_acc=("max_acc", "max"),
                          n_gt60=("n_gt60", "sum"), n_gt55=("n_gt55", "sum"))
print(agg.round(3).to_string())