# -*- coding: utf-8 -*-
"""E6: clean bypass-depth test (supervised, synthetic cue on real Kronos token context; frozen Kronos-small; no RL, no orders).
m_t = [c_t, z_t], c iid +-1, z N(0,1). Target y_t = c_{t-L} for t >= L.
Variants: none (head sees only h; m not used anywhere), direct (m only to head), emb (gated bypass at embedding, 1 encoder width 512),
multis (gated bypass at all 8 layer inputs, encoder hidden width 64 -> params matched to emb), multi (8 sites, width 512).
All bypass variants also feed m to the head (DIRECT head), as in the RL design. Gate init -2.
Early stopping on val acc (eval every 100 steps, min 2000 steps, patience 1000, cap 4000); test acc at best-val weights.
Splits: train bars [0,i1), val [i1,i2), test [i2,N)."""
from __future__ import annotations
import argparse, json, time, sys, copy
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
ROOT = Path(r"E:\rwkv7"); sys.path.insert(0, str(ROOT / "kronos_src"))
from model import Kronos
OUT = ROOT / "runs" / "overnight_2026-10-18"; DATA = ROOT / "data" / "binance" / "klines_1m" / "bars_1m_of_3m.csv"
ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lag", type=int, required=True); ap.add_argument("--cap", type=int, default=4000); ap.add_argument("--min_steps", type=int, default=2000)
ap.add_argument("--patience", type=int, default=1000); ap.add_argument("--B", type=int, default=32); ap.add_argument("--T", type=int, default=96)
ap.add_argument("--tag", default="")
a = ap.parse_args(); dev = "cuda"; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); T0 = time.time()
df = pd.read_csv(DATA, usecols=["ts"]); N = len(df); i1, i2 = int(N * 0.70), int(N * 0.85)
model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
for p in model.parameters(): p.requires_grad = False
z = np.load(OUT / "e5_tokens.npz"); ts = pd.to_datetime(df["ts"], unit="s", utc=True)
STAMP = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1).astype(np.float32)
S1t, S2t, STt = torch.tensor(z["S1"], device=dev), torch.tensor(z["S2"], device=dev), torch.tensor(STAMP, device=dev)
H = model.d_model; NL = len(model.transformer)
sites = {"none": [], "direct": [], "emb": [0], "multis": list(range(NL)), "multi": list(range(NL))}[a.variant]
W = 64 if a.variant == "multis" else H
use_m_head = a.variant != "none"
def mlp(w): return nn.Sequential(nn.Linear(2, w), nn.GELU(), nn.Linear(w, H))
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc = nn.ModuleList([mlp(W) for _ in sites]); s.g = nn.Parameter(torch.full((len(sites),), -2.0))
        s.enc_head = mlp(H); s.pi = nn.Sequential(nn.Linear(2 * H, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(s, s1, s2, st, m):
        x = model.embedding([s1, s2]) + model.time_emb(st)
        for i, layer in enumerate(model.transformer):
            if i in sites:
                k = sites.index(i); u = s.enc[k](m); u = u / (u.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6); x = x + torch.sigmoid(s.g[k]) * u
            x = layer(x)
        h = model.norm(x); hm = s.enc_head(m)
        if not use_m_head: hm = torch.zeros_like(hm)
        return s.pi(torch.cat([h / (h.norm(dim=-1, keepdim=True) + 1e-6) * 8, hm], -1)).squeeze(-1)
net = Net().to(dev)
params = [p for n_, p in net.named_parameters() if use_m_head or not n_.startswith("enc_head")]
opt = torch.optim.Adam(params, lr=1e-3); lossf = nn.BCEWithLogitsLoss()
def batch(lo, hi, B, g):
    st = g.integers(lo, hi - a.T, B); idx = torch.tensor(st[:, None] + np.arange(a.T), device=dev)
    c = torch.tensor(g.choice([-1.0, 1.0], size=(B, a.T)).astype(np.float32), device=dev)
    zz = torch.tensor(g.standard_normal((B, a.T)).astype(np.float32), device=dev)
    y = (c[:, :a.T - a.lag] > 0).float()
    return S1t[idx], S2t[idx], STt[idx], torch.stack([c, zz], -1), y
vg = np.random.default_rng(999); VAL = [batch(i1, i2, 64, vg) for _ in range(4)]
tg = np.random.default_rng(777); TEST = [batch(i2, N, 64, tg) for _ in range(8)]
def evaluate(S):
    net.eval(); cor = n = 0
    with torch.no_grad():
        for s1, s2, st, m, y in S:
            lg = net(s1, s2, st, m)[:, a.lag:]; cor += ((lg > 0).float() == y).sum().item(); n += y.numel()
    net.train(); return cor / n
curve = []; first60 = None; best = -1; best_step = 0; best_state = None; stop_reason = "cap"
for it in range(1, a.cap + 1):
    s1, s2, st, m, y = batch(0, i1, a.B, rng)
    lg = net(s1, s2, st, m)[:, a.lag:]; loss = lossf(lg, y); opt.zero_grad(); loss.backward(); opt.step()
    if it % 100 == 0:
        acc = evaluate(VAL); curve.append({"step": it, "val_acc": acc, "loss": float(loss), "gates": torch.sigmoid(net.g).tolist()})
        if first60 is None and acc >= 0.60: first60 = it
        if acc > best + 0.001: best, best_step, best_state = acc, it, copy.deepcopy(net.state_dict())
        print(json.dumps(curve[-1]), flush=True)
        if it >= a.min_steps and it - best_step >= a.patience: stop_reason = "early_stop"; break
net.load_state_dict(best_state); test_acc = evaluate(TEST)
ntr = sum(p.numel() for p in params)
res = {"variant": a.variant, "seed": a.seed, "lag": a.lag, "enc_width": W, "sites": sites, "trainable_params": int(ntr),
       "test_acc": test_acc, "best_val_acc": best, "best_step": best_step, "steps_run": it, "stop_reason": stop_reason,
       "steps_to_60_val": first60, "gates_at_best": [c["gates"] for c in curve if c["step"] == best_step][0], "curve": curve,
       "n_val_targets": int(sum(v[4].numel() for v in VAL)), "n_test_targets": int(sum(v[4].numel() for v in TEST)),
       "runtime_s": round(time.time() - T0, 1)}
json.dump(res, open(OUT / f"e6{a.tag}_{a.variant}_L{a.lag}_s{a.seed}.json", "w"), indent=1)
print("DONE", json.dumps({k: res[k] for k in ["variant", "seed", "lag", "trainable_params", "test_acc", "best_step", "steps_run", "stop_reason", "steps_to_60_val", "runtime_s"]}))
