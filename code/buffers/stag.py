# -*- coding: utf-8 -*-
"""Staggered / 冗余对照 (CPU, frozen Kronos-small, 零训练线性探针)
问题: "multis > emb" 是因为"层 = hop"(每层推进一个时间步), 还是仅仅因为多写几个位置 = 冗余容量?
做法: 在层 i 的输入注入已知 bit 的移位副本 c_{t-d_i} * w_i (w_i 随机单位向量, ||w|| = a*mean||x_i||),
     走完真实 8 层后在最后残差 h (model.norm 后) 上用岭回归探针读 c_{t-L}。
臂:
  all_same   : sites {0..7}, d_i = L            (E6 multis 的"同 lag 全层写")
  staggered  : sites {0..7}, d_i = L-i          (每深一层写更近的 token; 层=hop 的写法)
  site_0_7   : sites {0,7},  d_i = L
  site_0     : sites {0}
  site_7     : sites {7}
  none       : 不注入 (0.5 对照)
  all_wrong  : sites {0..7}, d_i = L+8 (内容错位对照, 应 = 0.5)
另外: L=0 正控; L=10 的 all_same / site_0_7 / site_0。
落盘: stag.json
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(r"E:\rwkv7"); sys.path.insert(0, str(ROOT / "kronos_src"))
from model import Kronos

dev = "cpu"; torch.set_num_threads(8)
OUT = Path(r"C:\Users\lele\AppData\Local\Temp\opencode\stag.json")
T, B, NB = 96, 4, 4
A = 1.0
SEEDS = [0, 1, 2]

df = pd.read_csv(ROOT / "data" / "binance" / "klines_1m" / "bars_1m_of_3m.csv", usecols=["ts"])
N = len(df); z = np.load(ROOT / "runs" / "e6_long_2026-10-20" / "e5_tokens.npz")
ts = pd.to_datetime(df["ts"], unit="s", utc=True)
STAMP = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1).astype(np.float32)
model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
H = model.d_model; NL = len(model.transformer)


def make_batches(seed):
    rng = np.random.default_rng(200 + seed)
    out = []
    for _ in range(NB):
        st = rng.integers(int(N * 0.85), N - T, B)
        idx = torch.tensor(st[:, None] + np.arange(T))
        out.append((torch.tensor(z["S1"][idx.numpy()]),
                    torch.tensor(z["S2"][idx.numpy()]),
                    torch.tensor(STAMP[idx.numpy()]),
                    torch.tensor(rng.choice([-1.0, 1.0], size=(B, T)).astype(np.float32))))
    return out


def shift(c, d):
    if d <= 0: return c.clone() if d == 0 else torch.zeros_like(c)
    o = torch.zeros_like(c); o[:, d:] = c[:, :T - d]; return o


def run_arm(batches, w, scale, sites, delays, probe_dir, a=A):
    """返回 [(h, c)]"""
    outs = []
    with torch.no_grad():
        for (s1, s2, st, c) in batches:
            x = model.embedding([s1, s2]) + model.time_emb(st)
            for i, layer in enumerate(model.transformer):
                if i in sites:
                    d = delays[i]
                    x = x + shift(c, d)[:, :, None] * (w[i] * (a * scale[i]))[None, None, :]
                x = layer(x)
            h = model.norm(x)
            h64 = h @ probe_dir
            outs.append((h64.numpy(), h.numpy(), c.numpy()))
    return outs


def ridge_probe(Xtr, ytr, Xte, yte, lam):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    A_ = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]
    Bm = np.c_[np.ones(len(Xte)), (Xte - mu) / sd]
    M = A_.T @ A_ + lam * np.eye(A_.shape[1]); M[0, 0] -= lam
    beta = np.linalg.solve(M, A_.T @ ytr)
    return float((np.sign(Bm @ beta) == yte).mean())


def probe(outs, L, which, lam):
    Xtr, ytr, Xte, yte = [], [], [], []
    for bi, (h64, hfull, c) in enumerate(outs):
        F = h64 if which == "p64" else hfull
        feats = F[:, L:, :].reshape(-1, F.shape[-1])
        lab = c[:, :T - L].reshape(-1)
        if bi < 2: Xtr.append(feats); ytr.append(lab)
        else: Xte.append(feats); yte.append(lab)
    return ridge_probe(np.concatenate(Xtr), np.concatenate(ytr),
                       np.concatenate(Xte), np.concatenate(yte), lam)


rows = []
t0 = time.time()
PROJ = torch.tensor(np.random.default_rng(5).standard_normal((H, 64)).astype(np.float32)) / np.sqrt(H)
for seed in SEEDS:
    batches = make_batches(seed)
    # 每层激活范数 (无注入)
    scale = {}
    with torch.no_grad():
        for i in range(NL):
            x = model.embedding([batches[0][0], batches[0][1]]) + model.time_emb(batches[0][2])
            for j in range(i): x = model.transformer[j](x)
            scale[i] = float(x.flatten(1).norm(dim=1).mean())
    w = [torch.randn(H) / torch.linalg.norm(torch.randn(H)) for _ in range(NL)]
    w = [None] + w[1:]  # 固定 seed 索引无关, 保持列表可读
    for i in range(NL):
        torch.manual_seed(1000 * seed + i)
        v = torch.randn(H); w[i] = v / v.norm()

    arms = [
        ("all_same_L15", list(range(8)), lambda i: 15, 15),
        ("staggered_L15", list(range(8)), lambda i: 15 - i, 15),
        ("site_0_7_L15", [0, 7], lambda i: 15, 15),
        ("site_0_L15", [0], lambda i: 15, 15),
        ("site_7_L15", [7], lambda i: 15, 15),
        ("none_L15", [], lambda i: 15, 15),
        ("all_current_L15", list(range(8)), lambda i: 0, 15),
        ("staggered_cur_L15", list(range(8)), lambda i: i, 15),
        ("staggered_cur_L7", list(range(8)), lambda i: min(i, 7), 7),
        ("all_wrong_L23", list(range(8)), lambda i: 23, 15),
        ("all_same_L0", list(range(8)), lambda i: 0, 0),
        ("all_same_L10", list(range(8)), lambda i: 10, 10),
        ("site_0_7_L10", [0, 7], lambda i: 10, 10),
        ("site_0_L10", [0], lambda i: 10, 10),
        ("site_7_L10", [7], lambda i: 10, 10),
        ("none_L10", [], lambda i: 10, 10),
    ]
    for tag, sites, dfun, L in arms:
        delays = {i: dfun(i) for i in range(NL)}
        outs = run_arm(batches, w, scale, sites, delays, PROJ)
        acc64 = probe(outs, L, "p64", 100.0)
        accf = probe(outs, L, "full", 1000.0)
        rows.append({"seed": seed, "arm": tag, "L": L, "sites": sites,
                     "delays": [delays[i] for i in sites],
                     "acc_proj64": round(acc64, 3), "acc_full512": round(accf, 3),
                     "amp": A})
        print(f"[s{seed}] {tag:<16} acc64={acc64:.3f} acc512={accf:.3f}", flush=True)

r = pd.DataFrame(rows)
r.to_csv(str(OUT).replace(".json", ".csv"), index=False)
json.dump({"meta": {"script": "stag.py", "T": T, "B": B, "NB": NB, "amp": A,
                    "seeds": SEEDS, "readout": "ridge on model.norm(x) (64-d random proj + full 512)",
                    "train_batches": 2, "test_batches": 2,
                    "runtime_s": round(time.time() - t0, 1)},
           "rows": rows}, open(OUT, "w"), indent=1)

print("\n=== mean over seeds ===")
print(r.pivot_table(index=["L", "arm"], values=["acc_proj64", "acc_full512"],
                    aggfunc="mean").round(3).to_string())
print(f"\nsaved {OUT} ({time.time()-t0:.0f}s)")