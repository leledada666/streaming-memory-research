# -*- coding: utf-8 -*-
"""RWKV7 state-recall probe (inference only, frozen weights, no training of the backbone).
Question: can a linear map read a bit written L steps ago from state_t?
Protocol (matches E6 synthetic-recall style but injected at input embedding):
  x_t = EMB(BOS) + alpha * c_t * w   (w fixed unit vector, c_t = +-1 iid)
  state_t = RWKV7 recurrent state after consuming token t = past[i]['recurrent_state'] [B,H,D,D]
  probe: ridge  state_t (per-layer sparse-ish random projection) -> sign(c_{t-L})
Controls: L=0 (positive ceiling), random w per step (must fail), alpha sweep.
Outputs JSON + npz under runs/rwkv_state_probe/.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

ROOT = Path(r"E:\rwkv7"); OUT = ROOT / "runs" / "rwkv_state_probe"; OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=768); ap.add_argument("--prime", type=int, default=64)
ap.add_argument("--layers", default="0,4,9,14,19,23"); ap.add_argument("--pdim", type=int, default=128)
ap.add_argument("--alphas", default="0.25,1.0"); ap.add_argument("--randw", type=int, default=1)
ap.add_argument("--lags", default="0,1,5,10,15,30,60"); ap.add_argument("--tag", default="smoke")
a = ap.parse_args()
layers = [int(x) for x in a.layers.split(",")]; alphas = [float(x) for x in a.alphas.split(",")]
lags = [int(x) for x in a.lags.split(",")]
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); rng = np.random.default_rng(0); T0 = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL = ROOT / "models" / "models" / "RWKV--RWKV7-Goose-World2.9-0.4B-HF" / "snapshots" / "master"
if not MODEL.exists(): MODEL = ROOT / "models" / "RWKV7-Goose-World2.9-0.4B-HF"
bb = AutoModelForCausalLM.from_pretrained(str(MODEL), trust_remote_code=True, torch_dtype=torch.float32).to(dev).eval()
for p in bb.parameters(): p.requires_grad = False
rtok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
H = int(bb.config.hidden_size); BOS = int(getattr(bb.config, "bos_token_id", 1) or 1)
EMB = bb.get_input_embeddings()
mean_norm = float(EMB.weight.detach().norm(dim=-1).median())
print(json.dumps({"dev": dev, "hidden": H, "bos": BOS, "emb_median_norm": mean_norm,
                  "n_layers": int(bb.config.num_hidden_layers), "layers": layers}), flush=True)

w0 = torch.randn(H, generator=torch.Generator().manual_seed(7))
w0 = (w0 / w0.norm()).to(dev)                     # fixed write direction, unit norm
SDIM = int(bb.config.value_dim[0]) * int(bb.config.head_dim)   # 16 heads x 64 x 64 = 65536
PROJ = {li: torch.randn(a.pdim, SDIM, device=dev) / np.sqrt(SDIM) for li in layers}  # fixed random read-out
print(json.dumps({"state_dim_per_layer": SDIM}), flush=True)
PAD = torch.zeros(a.pdim, device=dev)
fwd = bb.model

@torch.no_grad()
def run(alpha: float, random_w: bool, n: int):
    c = (rng.standard_normal(n) > 0).astype(np.int64) * 2 - 1      # +-1 iid bits
    ct = torch.tensor(c.astype(np.float32), device=dev)[:, None, None]
    bos = torch.full((1, 1), BOS, dtype=torch.long, device=dev)
    e0 = EMB(bos)                                                  # neutral prime (no bit)
    o = fwd(inputs_embeds=e0.expand(1, a.prime, H), use_cache=True)
    past = o.past_key_values
    feats = {li: np.zeros((n, a.pdim), np.float32) for li in layers}
    norms = {li: np.zeros(n, np.float32) for li in layers}
    wcur = w0
    for t in range(n):
        if random_w:                                               # control: new direction every step
            wcur = torch.randn(H, device=dev); wcur = wcur / wcur.norm()
        x = EMB(bos) + alpha * mean_norm * ct[t].view(1, 1, 1) * wcur.view(1, 1, H)
        o = fwd(inputs_embeds=x, past_key_values=past, use_cache=True); past = o.past_key_values
        for li in layers:
            S = past[li]["recurrent_state"]                        # [1, heads, D, D]
            S = S.reshape(-1)
            feats[li][t] = (PROJ[li] @ S).float().cpu().numpy()
            norms[li][t] = float(S.abs().mean())
    return c, feats, norms

def probe(X, y, frac=0.6):
    n = len(y); k = int(n * frac)
    best = (-1.0, None, 0.0)
    for al in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        m = Ridge(alpha=al).fit(X[:k], y[:k])
        v = roc_auc_score(y[k:], m.predict(X[k:]))
        tr = roc_auc_score(y[:k], m.predict(X[:k]))
        if v > best[0]: best = (v, al, tr)
    return {"auc_test": float(best[0]), "alpha": best[1], "auc_train": float(best[2])}

res = {"config": vars(a), "mean_norm": mean_norm, "runs": []}
for alpha in alphas:
    for random_w in ([False, True] if (a.randw and alpha == max(alphas)) else [False]):
        c, feats, norms = run(alpha, random_w, a.n)
        name = f"a{alpha}_{'randw' if random_w else 'fixedw'}"
        print(json.dumps({"pass": name, "elapsed_s": round(time.time() - T0, 1)}), flush=True)
        rec = {"pass": name, "alpha": alpha, "random_w": random_w, "rows": []}
        for li in layers:
            X = feats[li]
            for L in lags:
                if L >= len(c) - 20: continue
                y = ((c[:len(c) - L] > 0) if L > 0 else (c > 0)).astype(int)
                Xs = X[L:] if L > 0 else X
                r = probe(Xs, y); r.update({"layer": li, "lag": L})
                rec["rows"].append(r); print(json.dumps(r), flush=True)
        # all-layer concat (best single read-out)
        XA = np.concatenate([feats[li] for li in layers], 1)
        for L in lags:
            if L >= len(c) - 20: continue
            y = ((c[:len(c) - L] > 0) if L > 0 else (c > 0)).astype(int)
            Xs = XA[L:] if L > 0 else XA
            r = probe(Xs, y); r.update({"layer": -1, "lag": L}); rec["rows"].append(r)
        rec["state_meanabs_by_layer"] = {str(li): float(norms[li].mean()) for li in layers}
        rec["state_norm_source"] = "mean |S| over [heads, D, D]"
        res["runs"].append(rec)
        np.savez(OUT / f"feats_{a.tag}_{name}.npz", c=c, **{f"L{li}": feats[li] for li in layers},
                 **{f"n{li}": norms[li] for li in layers})

res["runtime_s"] = round(time.time() - T0, 1)
json.dump(res, open(OUT / f"probe_{a.tag}.json", "w"), indent=1)
print("DONE", res["runtime_s"], flush=True)