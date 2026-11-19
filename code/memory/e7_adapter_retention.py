# -*- coding: utf-8 -*-
"""E7 smoke (CPU only): does a trained memory adapter degrade the frozen Kronos-small backbone?

Backbone probes on h_t (frozen, no adapter):
  MAIN  RV5 : ridge predicts log realized variance of the next 5 one-minute bars; metric = OOS R2 (test split)
  MINOR DIR : logistic predicts sign of the next 1-minute return; metric = AUC (expected ~0.5, sanity only)
Adapter: gated GRU summary over a per-bar memory stream m_t (2 ch) -> Linear -> d_model,
         injected once at the embedding as bypass_delta = sigmoid(gate) * proj(GRU(m)).
Training: synthetic recall, y_t = c_{t-LAG}, m_t = [c_t, z_t] (c iid +-1): adapter must carry a past cue.
Conditions probed on the SAME anchors with the SAME protocol:
  base      h with no adapter
  gate0     trained adapter, gate -> 0 (must reproduce base: identity sanity)
  randinit  adapter, random init, never trained, gate init sigmoid(-2)=0.12
  trained   trained adapter fed the REAL memory stream (r, logvol)
  zeroin    trained adapter fed an all-zero memory stream
Each condition is scored twice: (i) probe REFIT on its own train features, (ii) BASE probe TRANSFERRED
(PCA/mean/std/weights from base) -> paired measure of representation drift.
No GPU, no orders, read-only inputs, writes only under Temp.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score

ROOT = Path(r"E:\rwkv7")
sys.path.insert(0, str(ROOT / "kronos_src"))
from model import Kronos

DATA = ROOT / "data" / "binance" / "klines_1m" / "bars_1m_of_3m.csv"
TOK = ROOT / "runs" / "overnight_2026-10-18" / "e5_tokens.npz"
OUT = Path(r"C:\Users\lele\AppData\Local\Temp\opencode\e7")

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, default=32)
ap.add_argument("--B", type=int, default=4)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--lag", type=int, default=8)
ap.add_argument("--ntr", type=int, default=3000)
ap.add_argument("--nva", type=int, default=800)
ap.add_argument("--nte", type=int, default=1200)
ap.add_argument("--cb", type=int, default=32)
ap.add_argument("--pca", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()

torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed); T0 = time.time()
OUT.mkdir(parents=True, exist_ok=True)
log = open(OUT / "smoke.log", "w", encoding="utf-8")
def P(*x):
    s = " ".join(str(i) for i in x); print(s, flush=True); log.write(s + "\n"); log.flush()

df = pd.read_csv(DATA)
close = df["close"].to_numpy(np.float64); N = len(close)
r = np.zeros(N); r[1:] = np.diff(np.log(close)) * 1e4
cs = np.concatenate([[0.0], np.cumsum(r ** 2)])
i1, i2 = int(N * 0.70), int(N * 0.85)
logrv = np.full(N, np.nan); logrv[: N - 5] = 0.5 * np.log(np.maximum(cs[6:] - cs[1: N - 4], 1e-6))
trail = np.full(N, np.nan); trail[5:] = 0.5 * np.log(np.maximum(cs[6:] - cs[1: N - 4], 1e-6))
diry = np.full(N, np.nan); diry[: N - 1] = (r[1:] > 0).astype(np.float64)
P(f"N={N} split={i1}/{i2 - i1}/{N - i2} T={a.T} B={a.B} steps={a.steps} lag={a.lag} pca={a.pca}")

model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).eval()
for p in model.parameters(): p.requires_grad_(False)
D = model.d_model
z = np.load(TOK); ts = pd.to_datetime(df["ts"], unit="s", utc=True)
STAMP = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1).astype(np.float32)
S1t, S2t, STt = torch.tensor(z["S1"]), torch.tensor(z["S2"]), torch.tensor(STAMP)
P(f"backbone d_model={D} layers={len(model.transformer)}")

rr = np.clip(r / (np.std(r[:i1]) + 1e-9), -3, 3)
lv = np.log(np.maximum(df["base_vol"].to_numpy(np.float64), 1e-9))
lv = np.clip((lv - lv[:i1].mean()) / (lv[:i1].std() + 1e-9), -3, 3)
MREAL = np.stack([rr, lv], 1).astype(np.float32)

def anchors(lo, hi, n):
    ok = np.arange(max(lo + a.T, 6), hi - 5)
    ok = ok[~np.isnan(logrv[ok]) & ~np.isnan(diry[ok]) & ~np.isnan(trail[ok])]
    return np.sort(rng.choice(ok, size=min(n, len(ok)), replace=False))
TR, VA, TE = anchors(a.T, i1, a.ntr), anchors(i1, i2, a.nva), anchors(i2, N, a.nte)
P(f"anchors train/val/test = {len(TR)}/{len(VA)}/{len(TE)}")

class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(2, 64, batch_first=True)
        self.proj = nn.Linear(64, D)
        self.gate = nn.Parameter(torch.tensor(-2.0))
    def delta(self, m, gate_scale=None):
        h, _ = self.gru(m)
        return self.proj(h) * (gate_scale if gate_scale is not None else torch.sigmoid(self.gate))

def win_idx(an):
    return torch.tensor(an[:, None] - a.T + 1 + np.arange(a.T))

def extract_h(anchor, m=None, adp=None, gate_scale=None):
    out = np.zeros((len(anchor), D), np.float32)
    with torch.no_grad():
        for b0 in range(0, len(anchor), a.cb):
            ix = win_idx(anchor[b0: b0 + a.cb])
            dl = adp.delta(torch.tensor(m[ix.numpy()]), gate_scale) if adp is not None else None
            h = model.forward_hidden(S1t[ix], S2t[ix], stamp=STt[ix], bypass_delta=dl)[:, -1]
            out[b0: b0 + len(ix)] = (h / (h.norm(dim=-1, keepdim=True) + 1e-6) * 8.0).numpy()
    return out

# ---------------- probes ----------------
ALPHAS = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
CS = [1e-4, 1e-3, 1e-2]

class Probe:
    """PCA(pca) -> standardize -> ridge RV / logistic DIR, fit on train, alpha chosen on val."""
    def fit(self, Xtr, Xva, kind):
        self.kind = kind
        self.pca = PCA(n_components=min(a.pca, Xtr.shape[0], Xtr.shape[1]), random_state=0).fit(Xtr)
        A, B = self.pca.transform(Xtr), self.pca.transform(Xva)
        self.mu, self.sd = A.mean(0), A.std(0) + 1e-6
        A, B = (A - self.mu) / self.sd, (B - self.mu) / self.sd
        if kind == "rv":
            ytr, yva = logrv[TR], logrv[VA]
            best = None
            for al in ALPHAS:
                m = Ridge(alpha=al).fit(A, ytr); v = r2_score(yva, m.predict(B))
                if best is None or v > best[0]: best = (v, al, m)
            self.valscore, self.hp, self.m = best
        else:
            ytr, yva = diry[TR].astype(int), diry[VA].astype(int)
            best = None
            for cc in CS:
                m = LogisticRegression(C=cc, max_iter=500).fit(A, ytr)
                v = roc_auc_score(yva, m.predict_proba(B)[:, 1])
                if best is None or v > best[0]: best = (v, cc, m)
            self.valscore, self.hp, self.m = best
        return self
    def X(self, X):
        return (self.pca.transform(X) - self.mu) / self.sd
    def score(self, Xte, yte):
        if self.kind == "rv":
            return float(r2_score(yte, self.m.predict(self.X(Xte))))
        p = self.m.predict_proba(self.X(Xte))[:, 1]
        return float(roc_auc_score(yte, p))

def run_all(H, tag, base=None):
    out = {}
    for kind in ("rv", "dir"):
        pr = Probe().fit(H["tr"], H["va"], kind)
        yte = logrv[TE] if kind == "rv" else diry[TE].astype(int)
        s_refit = pr.score(H["te"], yte)
        s_trans = base[kind].score(H["te"], yte) if base is not None else None
        out[kind] = {"hp": float(pr.hp), "val": float(pr.valscore), "refit": s_refit, "transfer": s_trans}
    P(json.dumps({"cond": tag, "rv_refit": round(out["rv"]["refit"], 4), "rv_transfer":
                  None if out["rv"]["transfer"] is None else round(out["rv"]["transfer"], 4),
                  "rv_alpha": out["rv"]["hp"], "dir_refit": round(out["dir"]["refit"], 4),
                  "dir_transfer": None if out["dir"]["transfer"] is None else round(out["dir"]["transfer"], 4)}))
    return out

t0 = time.time()
Hbase = {k: extract_h(an) for k, an in [("tr", TR), ("va", VA), ("te", TE)]}
P(f"[extract] base {time.time() - t0:.1f}s")
BASE = run_all(Hbase, "base")
BP = {"rv": Probe().fit(Hbase["tr"], Hbase["va"], "rv"), "dir": Probe().fit(Hbase["tr"], Hbase["va"], "dir")}
Xf = {k: np.stack([np.stack([trail[an - l] for l in range(1, 6)], 0) for an in v], 0).astype(np.float32)
      for k, v in [("tr", TR), ("va", VA), ("te", TE)]}
pr = Probe().fit(Xf["tr"], Xf["va"], "rv")
P(json.dumps({"cond": "ref_trailing_rv5", "rv_refit": round(pr.score(Xf["te"], logrv[TE]), 4),
              "rv_alpha": pr.hp, "rv_val": round(pr.valscore, 4)}))

# ---------------- train adapter (synthetic recall) ----------------
adp = Adapter()
enc_m = nn.Sequential(nn.Linear(2, D), nn.GELU(), nn.Linear(D, D))
headw = nn.Linear(2 * D, 1)
optA = torch.optim.Adam(adp.parameters(), lr=3e-3)
optH = torch.optim.Adam(list(enc_m.parameters()) + list(headw.parameters()), lr=1e-3)
lossf = nn.BCEWithLogitsLoss()

def train_batch(lo, hi, B, g):
    st = g.integers(lo + a.T, hi - 6, B)
    ix = torch.tensor(st[:, None] - a.T + 1 + np.arange(a.T))
    c = torch.tensor(g.choice([-1.0, 1.0], size=(B, a.T)).astype(np.float32))
    zz = torch.tensor(g.standard_normal((B, a.T)).astype(np.float32))
    return ix, torch.stack([c, zz], -1), c[:, : a.T - a.lag].gt(0).float()

vg = np.random.default_rng(999)
VALB = [train_batch(i1, i2, 16, vg) for _ in range(4)]
def acc_val():
    cor = n = 0
    with torch.no_grad():
        for ix, m, y in VALB:
            h = model.forward_hidden(S1t[ix], S2t[ix], stamp=STt[ix], bypass_delta=adp.delta(m))
            hn = h / (h.norm(dim=-1, keepdim=True) + 1e-6) * 8
            lg = headw(torch.cat([hn, enc_m(m)], -1)).squeeze(-1)[:, a.lag:]
            cor += ((lg > 0).float() == y).sum().item(); n += y.numel()
    return cor / n

P(f"adapter params={sum(p.numel() for p in adp.parameters())} gate0={float(torch.sigmoid(adp.gate)):.3f}")
t0 = time.time(); curve = []
for it in range(1, a.steps + 1):
    ix, m, y = train_batch(a.T, i1, a.B, rng)
    h = model.forward_hidden(S1t[ix], S2t[ix], stamp=STt[ix], bypass_delta=adp.delta(m))
    hn = h / (h.norm(dim=-1, keepdim=True) + 1e-6) * 8
    loss = lossf(headw(torch.cat([hn, enc_m(m)], -1)).squeeze(-1)[:, a.lag:], y)
    optA.zero_grad(); optH.zero_grad(); loss.backward(); optA.step(); optH.step()
    if it % 50 == 0:
        va = acc_val()
        curve.append({"step": it, "loss": round(float(loss), 4), "val_recall_acc": round(va, 3),
                      "gate_sigmoid": round(float(torch.sigmoid(adp.gate)), 4)})
        if it % 100 == 0: P(json.dumps(curve[-1]))
P(f"[adapter] {a.steps} steps in {time.time() - t0:.1f}s val_recall_acc={curve[-1]['val_recall_acc']:.3f}")

# ---------------- conditions ----------------
COND = {}
t0 = time.time()
hr = {k: extract_h(an, MREAL, adp) for k, an in [("tr", TR), ("va", VA), ("te", TE)]}
P(f"[extract] trained+real {time.time() - t0:.1f}s")
COND["trained_real"] = run_all(hr, "trained_real", BP)
t0 = time.time()
hz = {k: extract_h(an, np.zeros((N, 2), np.float32), adp) for k, an in [("tr", TR), ("va", VA), ("te", TE)]}
P(f"[extract] trained+zeroin {time.time() - t0:.1f}s")
COND["zeroin"] = run_all(hz, "zeroin", BP)
torch.manual_seed(a.seed + 7); adp_r = Adapter()
hrn = {k: extract_h(an, MREAL, adp_r) for k, an in [("tr", TR), ("va", VA), ("te", TE)]}
COND["randinit"] = run_all(hrn, "randinit", BP)
hg0 = {k: extract_h(an, MREAL, adp, gate_scale=0.0) for k, an in [("tr", TR), ("va", VA), ("te", TE)]}
COND["gate0"] = run_all(hg0, "gate0", BP)
g0diff = float(max(np.abs(hg0[k] - Hbase[k]).max() for k in ("tr", "va", "te")))

r0 = BASE["rv"]["refit"]; a0 = BASE["dir"]["refit"]
rows = []
for k in ("trained_real", "zeroin", "randinit", "gate0"):
    v = COND[k]
    rows.append({"cond": k, "rv_refit": v["rv"]["refit"], "dR2_rel_refit": (v["rv"]["refit"] - r0) / abs(r0),
                 "rv_transfer": v["rv"]["transfer"], "dR2_abs_transfer": v["rv"]["transfer"] - r0,
                 "dir_refit": v["dir"]["refit"], "dAUC_refit": v["dir"]["refit"] - a0})
res = {"config": vars(a), "N": N, "split": [i1, i2 - i1, N - i2], "anchors": [len(TR), len(VA), len(TE)],
       "base": BASE, "adapter_params": int(sum(p.numel() for p in adp.parameters())),
       "adapter_curve": curve, "rows": rows, "gate0_maxdiff_vs_base": g0diff,
       "runtime_s": round(time.time() - T0, 1)}
json.dump(res, open(OUT / "smoke_results.json", "w"), indent=1)

P("\n== summary ==")
P(f"base: RV5 OOS R2={r0:.4f} (refit) | DIR AUC={a0:.4f}")
for x in rows:
    P(f"{x['cond']:>13} R2_refit={x['rv_refit']:.4f} dR2_rel={x['dR2_rel_refit']:+.1%} "
      f"R2_transfer={x['rv_transfer']:.4f} dR2_trans={x['dR2_abs_transfer']:+.4f} "
      f"AUC={x['dir_refit']:.4f} dAUC={x['dAUC_refit']:+.4f}")
P(f"gate0 max|h'-h| = {g0diff:.3e} runtime {res['runtime_s']}s")
P("DONE")