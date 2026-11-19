# -*- coding: utf-8 -*-
"""Step 2: low-rank Delta-K / Delta-V KV-cache injection at Kronos layers 0-5.
Frozen W_k/W_v; dK = (MLP_k(m) @ U_k) * gate, dV similarly; written into each position's K/V so later tokens attend to them.
Same synthetic recall task as E6. Streaming consistency check vs full causal forward.
Target params ~663k (multis). Rank chosen so 6 layers * 2 * (2*r + r*d_k*n_heads approx) matches.
Kronos-small: d_model=512, n_heads=8, head_dim=64. We inject into attention by adding to projected K/V inside a custom forward.
"""
from __future__ import annotations
import argparse, json, time, sys, copy, math
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(r"E:\rwkv7"); sys.path.insert(0, str(ROOT / "kronos_src"))
from model import Kronos
OUT = ROOT / "runs" / "roadmap_2026-10-20"
DATA = ROOT / "data" / "binance" / "klines_1m" / "bars_1m_of_3m.csv"
TOK = ROOT / "runs" / "e6_long_2026-10-20" / "e5_tokens.npz"
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lag", type=int, required=True)
ap.add_argument("--cap", type=int, default=15000); ap.add_argument("--min_steps", type=int, default=2000)
ap.add_argument("--patience", type=int, default=800); ap.add_argument("--B", type=int, default=32); ap.add_argument("--T", type=int, default=96)
ap.add_argument("--rank", type=int, default=16); ap.add_argument("--sites", default="0,1,2,3,4,5")
ap.add_argument("--cons_only", type=int, default=0)
a = ap.parse_args(); sites = [int(x) for x in a.sites.split(",") if x!=""]
dev="cuda"; torch.manual_seed(a.seed); rng=np.random.default_rng(a.seed); T0=time.time()
df=pd.read_csv(DATA, usecols=["ts"]); N=len(df); i1,i2=int(N*0.70),int(N*0.85)
model=Kronos.from_pretrained(str(ROOT/"models"/"Kronos-small")).to(dev).eval()
for p in model.parameters(): p.requires_grad=False
z=np.load(TOK); ts=pd.to_datetime(df["ts"], unit="s", utc=True)
STAMP=np.stack([ts.dt.minute,ts.dt.hour,ts.dt.weekday,ts.dt.day,ts.dt.month],1).astype(np.float32)
S1t=torch.tensor(z["S1"],device=dev); S2t=torch.tensor(z["S2"],device=dev); STt=torch.tensor(STAMP,device=dev)
H=model.d_model; NL=len(model.transformer); n_heads=model.transformer[0].attn.n_heads if hasattr(model.transformer[0],'attn') else 8
# discover attn module structure
layer0=model.transformer[0]
attn_mod=None
for name,mod in layer0.named_children():
    if 'attn' in name.lower() or hasattr(mod,'q_proj') or hasattr(mod,'qkv') or hasattr(mod,'c_attn'):
        attn_mod=mod; break
if attn_mod is None:
    # fallback: print structure once
    print("LAYER0", layer0); print(list(layer0.named_children())[:20]); raise SystemExit("cannot find attn")
print("ATTN", type(attn_mod).__name__, [n for n,_ in attn_mod.named_children()])

# We implement injection by monkey-patching each selected layer's forward: after attn projects K,V, add dK,dV.
# Safer portable path: replace layer forward with a wrapper that runs original but intercepts via hook on K/V tensors.
# Practical approach for Kronos: run embedding+layers manually; for each site, compute residual on x (like multi) PLUS
# add low-rank contribution into the attention by modifying the layer input with a dual path that concatenates a
# virtual prefix... Simpler faithful design matching the brief:
#   At layer i in sites: u = MLP(m); dK = gate * (u @ U_k) reshaped to (B,T,n_heads,d_k); same for dV.
#   We patch Kronos Attention.forward to accept optional dk,dv and add them to k,v before SDPA.

# Inspect attention forward source keys
import inspect
src = inspect.getsource(attn_mod.forward)
print("FWD_LEN", len(src)); print(src[:800])