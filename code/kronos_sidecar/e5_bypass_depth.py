# -*- coding: utf-8 -*-
"""E5 synthetic bypass-depth test (supervised, short; frozen Kronos-small; no RL, no orders).
Real Binance 1m Kronos tokens as background context; a synthetic cue c_t in {-1,+1} (iid) is injected each step together with a
N(0,1) distractor: m_t = [c_t, z_t]. Target at step t (t >= LAG): y_t = c_{t-LAG}  (needs 30-step memory).
Variants (same DIRECT head [h/||h||*8 ; enc_head(m_t)] everywhere):
  direct: no bypass (head sees only current m_t -> cannot know c_{t-LAG}; chance control)
  emb:    gated residual bypass at the embedding only (as in the RL scripts)
  multi:  gated residual bypass at the input of every transformer layer (8 sites, incl. embedding)
Gate logits init -2 (sigmoid 0.12) as in the RL design."""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
ROOT = Path(r"E:\rwkv7"); sys.path.insert(0, str(ROOT / "kronos_src"))
from model import Kronos, KronosTokenizer
OUT = ROOT / "runs" / "overnight_2026-10-18"; DATA = ROOT / "data" / "binance" / "klines_1m" / "bars_1m_of_3m.csv"
ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--lag", type=int, default=30); ap.add_argument("--B", type=int, default=32)
ap.add_argument("--T", type=int, default=96)
a = ap.parse_args(); dev = "cuda"; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); T0 = time.time()
df = pd.read_csv(DATA); N = len(df); i1, i2 = int(N * 0.70), int(N * 0.85)
ftok = OUT / "e5_tokens.npz"
model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
for p in model.parameters(): p.requires_grad = False
if not ftok.exists():
    feat = np.stack([df["open"], df["high"], df["low"], df["close"], df["base_vol"], df["quote_vol"]], 1).astype(np.float64)
    mu, sd = feat[:i1].mean(0), feat[:i1].std(0); xn = np.clip((feat - mu) / (sd + 1e-5), -5, 5).astype(np.float32)
    tok = KronosTokenizer.from_pretrained(str(ROOT / "models" / "Kronos-Tokenizer-base")).to(dev).eval()
    S1 = np.zeros(N, np.int64); S2 = np.zeros(N, np.int64); xt = torch.tensor(xn, device=dev)
    with torch.no_grad():
        for b0 in range(0, N, 256):
            idx = [list(range(max(0, t - 63), t + 1)) for t in range(b0, min(N, b0 + 256))]
            L = max(len(i) for i in idx); idx = [[i[0]] * (L - len(i)) + i for i in idx]
            z = tok.encode(xt[torch.tensor(idx, device=dev)], half=True)
            S1[b0:b0 + len(idx)] = z[0][:, -1].cpu().numpy(); S2[b0:b0 + len(idx)] = z[1][:, -1].cpu().numpy()
    np.savez(ftok, S1=S1, S2=S2); del tok
z = np.load(ftok); ts = pd.to_datetime(df["ts"], unit="s", utc=True)
STAMP = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1).astype(np.float32)
S1t, S2t, STt = torch.tensor(z["S1"], device=dev), torch.tensor(z["S2"], device=dev), torch.tensor(STAMP, device=dev)
H = model.d_model; NL = len(model.transformer)
sites = {"direct": [], "emb": [0], "multi": list(range(NL))}[a.variant]
def mlp(): return nn.Sequential(nn.Linear(2, H), nn.GELU(), nn.Linear(H, H))
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc = nn.ModuleList([mlp() for _ in sites]); s.g = nn.Parameter(torch.full((len(sites),), -2.0))
        s.enc_head = mlp(); s.pi = nn.Sequential(nn.Linear(2 * H, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(s, s1, s2, st, m):
        x = model.embedding([s1, s2]) + model.time_emb(st)
        for i, layer in enumerate(model.transformer):
            if i in sites:
                k = sites.index(i); u = s.enc[k](m); u = u / (u.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6); x = x + torch.sigmoid(s.g[k]) * u
            x = layer(x)
        h = model.norm(x)
        return s.pi(torch.cat([h / (h.norm(dim=-1, keepdim=True) + 1e-6) * 8, s.enc_head(m)], -1)).squeeze(-1)
net = Net().to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3); lossf = nn.BCEWithLogitsLoss()
def batch(lo, hi, B, g):
    st = g.integers(lo, hi - a.T, B); idx = torch.tensor(st[:, None] + np.arange(a.T), device=dev)
    c = torch.tensor(g.choice([-1.0, 1.0], size=(B, a.T)).astype(np.float32), device=dev)
    zz = torch.tensor(g.standard_normal((B, a.T)).astype(np.float32), device=dev)
    y = (c[:, :-a.lag] > 0).float()  # target for t >= lag
    return S1t[idx], S2t[idx], STt[idx], torch.stack([c, zz], -1), y
vg = np.random.default_rng(999); VAL = [batch(i1, i2, 64, vg) for _ in range(2)]
def evaluate():
    net.eval(); cor = n = 0
    with torch.no_grad():
        for s1, s2, st, m, y in VAL:
            lg = net(s1, s2, st, m)[:, a.lag:]; cor += ((lg > 0).float() == y).sum().item(); n += y.numel()
    net.train(); return cor / n
curve = []; first75 = None
for it in range(1, a.steps + 1):
    s1, s2, st, m, y = batch(0, i1, a.B, rng)
    lg = net(s1, s2, st, m)[:, a.lag:]; loss = lossf(lg, y); opt.zero_grad(); loss.backward(); opt.step()
    if it % 100 == 0 or it == 1:
        acc = evaluate(); curve.append({"step": it, "val_acc": acc, "loss": float(loss), "gates": torch.sigmoid(net.g).tolist()})
        if first75 is None and acc >= 0.75: first75 = it
        print(json.dumps(curve[-1]), flush=True)
res = {"variant": a.variant, "seed": a.seed, "lag": a.lag, "steps": a.steps, "B": a.B, "T": a.T, "sites": sites, "final_val_acc": curve[-1]["val_acc"],
       "best_val_acc": max(c["val_acc"] for c in curve), "steps_to_75": first75, "curve": curve, "runtime_s": round(time.time() - T0, 1),
       "n_val_targets": int(sum(v[4].numel() for v in VAL)), "trainable_params": int(sum(p.numel() for p in net.parameters()))}
json.dump(res, open(OUT / f"e5_{a.variant}_seed{a.seed}.json", "w"), indent=1); print("DONE", json.dumps({k: res[k] for k in ["variant", "seed", "final_val_acc", "steps_to_75", "runtime_s"]}))