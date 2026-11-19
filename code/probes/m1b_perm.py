# m1b_perm.py —— m1b_clean 的置换零分布 + cross-fit 审计
# 问题：L=10 的 .526 是 max-over-64-head-sites（8层×8头）的选择偏差吗？
# 设计：
#  (1) 固定注入方向 w、振幅、以及**用真实 c 算出的特征**（不改特征），只把 probe 的目标标签 c 打乱；
#      变体 A(within)：每个 batch 内独立打乱时间轴；变体 B(cross)：跨 batch 全局打乱。
#      这才是"选择偏差"零分布：特征里含的是原 c，目标换成与特征无关的 c'，任何 >.5 只能来自选择/过拟合。
#  (2) 200 次重复，逐 seed 取 64 个 head 站点的 max，再跨 seed 取均值，与真实统计量同构。
#  (3) cross-fit：head 选择 + 阈值选择只用 train 折（batch 0-3 内的 2 折），评估在 test 折（batch 4-7）；
#      并给出 cross-fit 流程自身的置换零分布（同一套选择规则在打乱标签下重跑）。
#  (4) L=0 正控：真实 acc≈.96（读数成功），置换后逐站点应回落 .5，否则管线泄漏。
import math, sys, time, os
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
T = 96; B = 12; NB = 8; LAGS = [0, 1, 5, 10, 15]
SEEDS = [0, 1, 2]; NPERM = 200; LAM = 10.0
if os.environ.get("SMOKE"):          # 冒烟测试：1 seed / 3 次置换 / 2 个 lag
    SEEDS = [0]; NPERM = 3; LAGS = [0, 10]
TR = list(range(4)); TE = list(range(4, 8))
FOLDS = [([0, 1], [2, 3]), ([2, 3], [0, 1])]
model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
H = model.d_model; NL = len(model.transformer)
NH = model.transformer[0].self_attn.n_heads; DH = model.transformer[0].self_attn.head_dim
SITES = [(li, h) for li in range(NL) for h in range(NH)]
print(f"d_model={H} layers={NL} heads={NH} head_dim={DH}  站点数={len(SITES)}  NPERM={NPERM}", flush=True)


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


def rows(f, lag, bids):
    """f:(NB,B,T,D)；取 lag 之后的时间步，标签对齐到 t-lag"""
    return np.concatenate([f[b, :, lag:, :].reshape(-1, f.shape[-1]) for b in bids], 0)


def labels(cs, lag, bids):
    """cs:(NB,B,T) -> 与 rows() 同序的标签向量"""
    return np.concatenate([cs[b, :, :T - lag].reshape(-1) for b in bids], 0)


def design(Xtr, Xte, lam=LAM):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    A = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd].astype(np.float32)
    Bm = np.c_[np.ones(len(Xte)), (Xte - mu) / sd].astype(np.float32)
    M = (A.T @ A).astype(np.float64)
    M += lam * np.eye(M.shape[0]); M[0, 0] -= lam
    return A, Bm, M


def acc_of(beta, Bm, y):
    return float((np.sign(Bm @ beta) == y).mean())


def perm_labels(cs, mode, rng):
    if mode == "within":          # 每个 batch 内独立打乱时间轴
        out = cs.copy()
        for b in range(cs.shape[0]):
            out[b] = cs[b][:, rng.permutation(cs.shape[2])]
        return out
    if mode == "cross":           # 跨 batch 全局打乱
        return rng.permutation(cs.reshape(-1)).reshape(cs.shape)
    raise ValueError(mode)


t0 = time.time()
perm_rows, cf_rows, real_rows = [], [], []
null_by_seed, cfnull_by_seed = {}, {}
gate_null_by_seed = {}

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
    cs = np.stack([c.numpy() for _, _, _, c in batches]).reshape(NB, B, T)

    # 目标标签零分布（逐 seed 固定，跨 seed 配对）
    label_sets, perms = {}, {}
    for mode in ("within", "cross"):
        prng = np.random.default_rng(1000 + seed)
        ps = [perm_labels(cs, mode, prng) for _ in range(NPERM)]
        perms[mode] = ps
        for L in LAGS:
            label_sets[(mode, L, "tr")] = np.stack([labels(p, L, TR) for p in ps], 1).astype(np.float32)
            label_sets[(mode, L, "te")] = np.stack([labels(p, L, TE) for p in ps], 1).astype(np.float32)
            label_sets[(mode, L, "A")] = np.stack([labels(p, L, [0, 1]) for p in ps], 1).astype(np.float32)
            label_sets[(mode, L, "B")] = np.stack([labels(p, L, [2, 3]) for p in ps], 1).astype(np.float32)

    feats, res, gfeat = [None] * NL, [None] * NL, [None] * NL
    for li in range(NL):
        attn = model.transformer[li].self_attn
        x_in = []
        for bi, (s1, s2, stamp, c) in enumerate(batches):
            with torch.no_grad():
                x = model.embedding([s1, s2]) + model.time_emb(stamp)
                for j in range(li):
                    x = model.transformer[j](x)
            x_in.append(x)
        sc = float(torch.cat([xx.reshape(-1, H) for xx in x_in]).std())
        amp = sc * math.sqrt(H)
        x_inj, ho, x_out = [], [], []
        for bi in range(NB):
            with torch.no_grad():
                xi = x_in[bi] + batches[bi][3][:, :, None] * (w * amp)[None, None, :]
                a, _ = head_outputs(xi, attn)
                xo = model.transformer[li](xi)
            x_inj.append(xi.numpy()); ho.append(a.numpy()); x_out.append(xo.numpy())
        feats[li] = np.stack([np.stack([ho[b][:, h, :, :] for b in range(NB)]) for h in range(NH)])
        res[li] = np.stack(x_out); gfeat[li] = np.stack(x_inj)

    real_all, cfnull = {}, {}
    for L in LAGS:
        te_acc = np.zeros((len(SITES), NPERM)); sel_acc = np.zeros((len(SITES), NPERM))
        nullmax = {m: np.zeros(NPERM) for m in ("within", "cross")}
        site_nulls = {m: [] for m in ("within", "cross")}
        real_site = []
        YA, YB = label_sets[("within", L, "A")], label_sets[("within", L, "B")]
        for si, (li, h) in enumerate(SITES):
            f = feats[li][h]
            Xtr = rows(f, L, TR); Xte = rows(f, L, TE)
            ytr = labels(cs, L, TR).astype(np.float64); yte = labels(cs, L, TE)
            A, Bm, M = design(Xtr, Xte)
            beta = np.linalg.solve(M, A.T.astype(np.float64) @ ytr)
            real_site.append(acc_of(beta, Bm, yte)); real_all[(L, (li, h))] = real_site[-1]
            P = np.linalg.solve(M, A.T.astype(np.float64)).astype(np.float32)
            for m in ("within", "cross"):
                S = Bm @ (P @ label_sets[(m, L, "tr")])
                accp = (np.sign(S) == label_sets[(m, L, "te")]).mean(0)
                nullmax[m] = np.maximum(nullmax[m], accp)
                site_nulls[m].append(float(accp.mean()))
                if m == "within":
                    te_acc[si] = accp
            # 内层 2 折：只在打乱标签下做 head 选择（与 cross-fit 同规则）
            fold_vals = []
            for trb, vab, YtrP, YvaP in [(0, 1, YA, YB), (1, 0, YB, YA)]:
                A2, Bm2, M2 = design(rows(f, L, FOLDS[trb][0]), rows(f, L, FOLDS[trb][1]))
                P2 = np.linalg.solve(M2, A2.T.astype(np.float64)).astype(np.float32)
                fold_vals.append((np.sign(Bm2 @ (P2 @ YtrP)) == YvaP).mean(0))
            sel_acc[si] = 0.5 * (fold_vals[0] + fold_vals[1])
        real_max = max(real_site)
        real_rows.append({"seed": seed, "L": L, "real_max_over_sites": real_max,
                          "real_site_mean": float(np.mean(real_site))})
        # cross-fit 流程自身的零分布
        best_p = np.argmax(sel_acc, 0)
        cfnull[L] = te_acc[best_p, np.arange(NPERM)]
        cfnull_by_seed.setdefault(L, []).append(cfnull[L])
        for m in ("within", "cross"):
            null_by_seed.setdefault((m, L), []).append(nullmax[m])
            perm_rows.append({"seed": seed, "L": L, "mode": m,
                              "null_site_mean": float(np.mean(site_nulls[m])),
                              "null_med": float(np.median(nullmax[m])),
                              "null_p95": float(np.percentile(nullmax[m], 95)),
                              "null_p99": float(np.percentile(nullmax[m], 99))})
        print(f"seed {seed} L={L}: real_max={real_max:.4f} site_mean={np.mean(real_site):.4f} "
              f"| null[within] med={np.median(nullmax['within']):.4f} p95={np.percentile(nullmax['within'],95):.4f} "
              f"site_mean={np.mean(site_nulls['within']):.4f} "
              f"| cfnull med={np.median(cfnull[L]):.4f} p95={np.percentile(cfnull[L],95):.4f} ({time.time()-t0:.0f}s)",
              flush=True)

    for L in LAGS:   # 残差流参考
        accs = []
        for li in range(NL):
            A, Bm, M = design(rows(res[li], L, TR), rows(res[li], L, TE))
            beta = np.linalg.solve(M, A.T.astype(np.float64) @ labels(cs, L, TR).astype(np.float64))
            accs.append(acc_of(beta, Bm, labels(cs, L, TE)))
        print(f"  seed {seed} residual L={L} best={max(accs):.4f}", flush=True)

    for li in range(NL):   # L=0 正控：注入后层输入
        A, Bm, M = design(rows(gfeat[li], 0, TR), rows(gfeat[li], 0, TE))
        P = np.linalg.solve(M, A.T.astype(np.float64)).astype(np.float32)
        real_g = acc_of(np.linalg.solve(M, A.T.astype(np.float64) @ labels(cs, 0, TR).astype(np.float64)),
                        Bm, labels(cs, 0, TE))
        for m in ("within", "cross"):
            S = Bm @ (P @ label_sets[(m, 0, "tr")])
            accp = (np.sign(S) == label_sets[(m, 0, "te")]).mean(0)
            gate_null_by_seed.setdefault(m, []).append(accp)
        print(f"  seed {seed} gate L=0 layer {li} real={real_g:.4f} "
              f"null_within={np.mean(gate_null_by_seed['within'][-1]):.4f}", flush=True)

    # ---- cross-fit（真实标签）：head 选择 + 阈值选择都只在 train 折 ----
    for L in LAGS:
        oof, sel = {}, {}
        for (li, h) in SITES:
            f = feats[li][h]; parts = []; a = []
            for trb, vab in FOLDS:
                A, Bm, M = design(rows(f, L, trb), rows(f, L, vab))
                beta = np.linalg.solve(M, A.T.astype(np.float64) @ labels(cs, L, trb).astype(np.float64))
                strn = (A @ beta).astype(np.float64); mu_s, sd_s = strn.mean(), strn.std() + 1e-9
                s = (Bm.astype(np.float64) @ beta - mu_s) / sd_s
                y = labels(cs, L, vab)
                parts.append((s, y)); a.append(float((np.sign(s) == y).mean()))
            oof[(li, h)] = parts; sel[(li, h)] = float(np.mean(a))
        best = max(sel, key=lambda k: sel[k])
        s_oof = np.concatenate([p[0] for p in oof[best]]); y_oof = np.concatenate([p[1] for p in oof[best]])
        grid = np.linspace(-1.0, 1.0, 81)
        d = float(grid[np.argmax([(np.sign(s_oof - g) == y_oof).mean() for g in grid])])
        f = feats[best[0]][best[1]]
        A, Bm, M = design(rows(f, L, TR), rows(f, L, TE))
        beta = np.linalg.solve(M, A.T.astype(np.float64) @ labels(cs, L, TR).astype(np.float64))
        strn = (A @ beta).astype(np.float64); mu_s, sd_s = strn.mean(), strn.std() + 1e-9
        s_te = (Bm.astype(np.float64) @ beta - mu_s) / sd_s; y_te = labels(cs, L, TE)
        cf_rows.append({"seed": seed, "L": L,
                        "crossfit_acc": float((np.sign(s_te - d) == y_te).mean()),
                        "crossfit_acc_thr0": float((np.sign(s_te) == y_te).mean()),
                        "sel_layer": best[0], "sel_head": best[1], "sel_val_acc": sel[best],
                        "thr_norm": d, "real_at_sel_site": real_all[(L, best)]})
        print(f"seed {seed} L={L}: crossfit={cf_rows[-1]['crossfit_acc']:.4f} "
              f"(thr0 {cf_rows[-1]['crossfit_acc_thr0']:.4f}) 选中 L{best[0]}H{best[1]} "
              f"val={sel[best]:.4f} thr={d:+.2f} ({time.time()-t0:.0f}s)", flush=True)
    del feats, res, gfeat, perms, label_sets

# ---------- 汇总 ----------
perm = pd.DataFrame(perm_rows); cf = pd.DataFrame(cf_rows); real = pd.DataFrame(real_rows)
perm.to_csv(r"C:\Users\lele\AppData\Local\Temp\opencode\m1b_perm_null.csv", index=False)
cf.to_csv(r"C:\Users\lele\AppData\Local\Temp\opencode\m1b_perm_crossfit.csv", index=False)
real.to_csv(r"C:\Users\lele\AppData\Local\Temp\opencode\m1b_perm_real.csv", index=False)

out = []
for L in LAGS:
    rl = real[real.L == L]; cfl = cf[cf.L == L]
    rec = {"L": L, "real_acc": rl.real_max_over_sites.mean(), "real_sd": rl.real_max_over_sites.std(ddof=1),
           "real_site_mean": rl.real_site_mean.mean(),
           "crossfit": cfl.crossfit_acc.mean(), "crossfit_sd": cfl.crossfit_acc.std(ddof=1),
           "crossfit_thr0": cfl.crossfit_acc_thr0.mean()}
    for m in ("within", "cross"):
        arr = np.mean(np.stack(null_by_seed[(m, L)]), 0)
        rec[f"null_{m}_med"] = float(np.median(arr))
        rec[f"null_{m}_p95"] = float(np.percentile(arr, 95))
        rec[f"null_{m}_p99"] = float(np.percentile(arr, 99))
        rec[f"p_{m}"] = float((1 + (arr >= rec["real_acc"]).sum()) / (NPERM + 1))
        rec[f"null_{m}_site_mean"] = perm[(perm.L == L) & (perm["mode"] == m)]["null_site_mean"].mean()
    cfa = np.mean(np.stack(cfnull_by_seed[L]), 0)
    rec["cfnull_med"] = float(np.median(cfa)); rec["cfnull_p95"] = float(np.percentile(cfa, 95))
    rec["cfnull_p99"] = float(np.percentile(cfa, 99))
    rec["p_cf"] = float((1 + (cfa >= rec["crossfit"]).sum()) / (NPERM + 1))
    out.append(rec)
summ = pd.DataFrame(out)
summ.to_csv(r"C:\Users\lele\AppData\Local\Temp\opencode\m1b_perm_summary.csv", index=False)

grand = {}
for m in ("within", "cross"):
    arr = np.maximum.reduce([np.mean(null_by_seed[(m, l)], 0) for l in LAGS])
    grand[m] = (float(np.percentile(arr, 95)), float(np.percentile(arr, 99)))
gate = {}
for m in ("within", "cross"):
    a = np.mean(np.stack(gate_null_by_seed[m]), 0)
    gate[m] = (float(a.mean()), float(np.percentile(a, 95)), float(a.max()))

print("\n=== 表：真实 vs 置换零分布 vs cross-fit（200 次置换，跨 seed 配对平均）===")
show = pd.DataFrame({
    "L": summ.L,
    "real_acc": summ.real_acc.map(lambda x: f"{x:.3f}"),
    "real_site_mean": summ.real_site_mean.map(lambda x: f"{x:.3f}"),
    "null_med": summ["null_within_med"].map(lambda x: f"{x:.3f}"),
    "null_p95": summ["null_within_p95"].map(lambda x: f"{x:.3f}"),
    "null_p99": summ["null_within_p99"].map(lambda x: f"{x:.3f}"),
    "null_site_mean": summ["null_within_site_mean"].map(lambda x: f"{x:.3f}"),
    "crossfit_acc": summ.crossfit.map(lambda x: f"{x:.3f}"),
    "cf_null_p95": summ.cfnull_p95.map(lambda x: f"{x:.3f}"),
    "p": summ["p_within"].map(lambda x: f"{x:.4f}"),
})
show["显著"] = ["是(p99)" if r > q else ("边际(p95)" if r > t else "否")
              for r, q, t in zip(summ.real_acc, summ["null_within_p99"], summ["null_within_p95"])]
show["cf显著"] = ["是" if r > q else "否" for r, q in zip(summ.crossfit, summ.cfnull_p95)]
print(show.to_string(index=False))
print("\n--- 跨 batch 打乱（cross）---")
print(pd.DataFrame({"L": summ.L,
                    "null_med": summ["null_cross_med"].map(lambda x: f"{x:.3f}"),
                    "null_p95": summ["null_cross_p95"].map(lambda x: f"{x:.3f}"),
                    "null_p99": summ["null_cross_p99"].map(lambda x: f"{x:.3f}"),
                    "p": summ["p_cross"].map(lambda x: f"{x:.4f}")}).to_string(index=False))
print(f"\nL 也参与选择（max over 5 lag × 64 站点）零分布 p95={grand['within'][0]:.3f} p99={grand['within'][1]:.3f}"
      f" ; cross p95={grand['cross'][0]:.3f} p99={grand['cross'][1]:.3f}")
print(f"L=0 正控（注入后层输入，逐站点置换）: within mean={gate['within'][0]:.4f} p95={gate['within'][1]:.4f} "
      f"max={gate['within'][2]:.4f} ; cross mean={gate['cross'][0]:.4f} max={gate['cross'][2]:.4f}")
print(f"cross-fit 选中站点分布: {cf.groupby(['sel_layer', 'sel_head']).size().to_dict()}")
print(f"总耗时 {time.time()-t0:.0f}s")