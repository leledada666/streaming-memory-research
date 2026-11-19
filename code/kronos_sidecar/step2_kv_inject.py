# -*- coding: utf-8 -*-
"""Step2: Delta-K/Delta-V low-rank injection into Kronos self-attn KV at layers 0-5.
Frozen W_k/W_v; after RoPE on current tokens, k += gate_k * (coeff_k @ U_k), same for v.
coeff = MLP(m) -> (B,T,n_heads,rank); U is (n_heads,rank,head_dim). DIRECT head included.
Param target ~663k (multis). Same synthetic recall as E6. Optional --cons_check.
"""
from __future__ import annotations
import argparse, json, time, sys, copy
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
ap.add_argument("--rank", type=int, default=12); ap.add_argument("--mid", type=int, default=128)
ap.add_argument("--sites", default="0,1,2,3,4,5"); ap.add_argument("--cons_check", type=int, default=0)
a = ap.parse_args(); sites = [int(x) for x in a.sites.split(",") if x != ""]
dev = "cuda"; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); T0 = time.time()
df = pd.read_csv(DATA, usecols=["ts"]); N = len(df); i1, i2 = int(N * 0.70), int(N * 0.85)
model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
for p in model.parameters(): p.requires_grad = False
z = np.load(TOK); ts = pd.to_datetime(df["ts"], unit="s", utc=True)
STAMP = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1).astype(np.float32)
S1t, S2t, STt = torch.tensor(z["S1"], device=dev), torch.tensor(z["S2"], device=dev), torch.tensor(STAMP, device=dev)
H = model.d_model; attn0 = model.transformer[0].self_attn; n_heads, head_dim = attn0.n_heads, attn0.head_dim

class Injector(nn.Module):
    def __init__(s):
        super().__init__()
        s.sites = sites
        s.mlp_k = nn.ModuleList([nn.Sequential(nn.Linear(2, a.mid), nn.GELU(), nn.Linear(a.mid, n_heads * a.rank)) for _ in sites])
        s.mlp_v = nn.ModuleList([nn.Sequential(nn.Linear(2, a.mid), nn.GELU(), nn.Linear(a.mid, n_heads * a.rank)) for _ in sites])
        s.U_k = nn.ParameterList([nn.Parameter(torch.randn(n_heads, a.rank, head_dim) * 0.02) for _ in sites])
        s.U_v = nn.ParameterList([nn.Parameter(torch.randn(n_heads, a.rank, head_dim) * 0.02) for _ in sites])
        s.g_k = nn.Parameter(torch.full((len(sites),), -2.0))
        s.g_v = nn.Parameter(torch.full((len(sites),), -2.0))
        s.enc_head = nn.Sequential(nn.Linear(2, H), nn.GELU(), nn.Linear(H, H))
        s.pi = nn.Sequential(nn.Linear(2 * H, 128), nn.GELU(), nn.Linear(128, 1))
        s._delta = {}
    def deltas(s, m):
        out = {}; B, T, _ = m.shape
        for i, li in enumerate(s.sites):
            ck = s.mlp_k[i](m).view(B, T, n_heads, a.rank).permute(0, 2, 1, 3)
            cv = s.mlp_v[i](m).view(B, T, n_heads, a.rank).permute(0, 2, 1, 3)
            dK = torch.sigmoid(s.g_k[i]) * torch.einsum("bhtr,hrd->bhtd", ck, s.U_k[i])
            dV = torch.sigmoid(s.g_v[i]) * torch.einsum("bhtr,hrd->bhtd", cv, s.U_v[i])
            out[li] = (dK, dV)
        return out
    def gates(s):
        return {"K": torch.sigmoid(s.g_k).detach().cpu().tolist(), "V": torch.sigmoid(s.g_v).detach().cpu().tolist()}

net = Injector().to(dev)

# Patch each selected self_attn.forward to add deltas after RoPE on current tokens (before cache cat)
_orig = {}
def install_patches():
    for li in sites:
        attn = model.transformer[li].self_attn
        if li in _orig: continue
        _orig[li] = attn.forward
        def make(li=li, attn=attn, orig=_orig[li]):
            def fwd(x, key_padding_mask=None, past_kv=None, use_cache=False):
                batch_size, seq_len, _ = x.shape
                q = attn.q_proj(x).view(batch_size, seq_len, attn.n_heads, attn.head_dim).transpose(1, 2)
                k = attn.k_proj(x).view(batch_size, seq_len, attn.n_heads, attn.head_dim).transpose(1, 2)
                v = attn.v_proj(x).view(batch_size, seq_len, attn.n_heads, attn.head_dim).transpose(1, 2)
                position_offset = 0 if past_kv is None else past_kv[0].shape[-2]
                q, k = attn.rotary(q, k, position_offset=position_offset)
                dlt = net._delta.get(li)
                if dlt is not None:
                    dK, dV = dlt
                    # dK/dV cover full sequence positions in non-cache mode; in cache mode only current tokens
                    if past_kv is None:
                        k = k + dK; v = v + dV
                    else:
                        k = k + dK[:, :, -seq_len:]; v = v + dV[:, :, -seq_len:]
                if past_kv is not None:
                    k = torch.cat([past_kv[0], k], dim=-2); v = torch.cat([past_kv[1], v], dim=-2)
                kv_len = k.shape[-2]
                if past_kv is None:
                    is_causal = True; attn_mask = None
                else:
                    is_causal = False
                    if seq_len > 1:
                        q_pos = torch.arange(position_offset, position_offset + seq_len, device=x.device)[:, None]
                        k_pos = torch.arange(kv_len, device=x.device)[None, :]
                        disallow = k_pos > q_pos
                        attn_mask = (~disallow)[None, None, :, :].expand(batch_size, attn.n_heads, -1, -1)
                    else:
                        attn_mask = None
                if key_padding_mask is not None and past_kv is None:
                    # keep none for our synthetic (no pad)
                    pass
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                    dropout_p=attn.attn_dropout_p if attn.training else 0.0, is_causal=is_causal)
                out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, attn.d_model)
                out = attn.resid_dropout(attn.out_proj(out))
                if use_cache: return out, (k, v)
                return out
            return fwd
        attn.forward = make()

install_patches()

def backbone(s1, s2, st, m):
    net._delta = net.deltas(m)
    x = model.embedding([s1, s2]) + model.time_emb(st)
    for layer in model.transformer:
        x = layer(x)
    h = model.norm(x)
    net._delta = {}
    return h

def logits(s1, s2, st, m):
    h = backbone(s1, s2, st, m)
    return net.pi(torch.cat([h / (h.norm(dim=-1, keepdim=True) + 1e-6) * 8, net.enc_head(m)], -1)).squeeze(-1)

def batch(lo, hi, B, g):
    st = g.integers(lo, hi - a.T, B); idx = torch.tensor(st[:, None] + np.arange(a.T), device=dev)
    c = torch.tensor(g.choice([-1.0, 1.0], size=(B, a.T)).astype(np.float32), device=dev)
    zz = torch.tensor(g.standard_normal((B, a.T)).astype(np.float32), device=dev)
    y = (c[:, : a.T - a.lag] > 0).float()
    return S1t[idx], S2t[idx], STt[idx], torch.stack([c, zz], -1), y

vg = np.random.default_rng(999); VAL = [batch(i1, i2, 64, vg) for _ in range(4)]
tg = np.random.default_rng(777); TEST = [batch(i2, N, 64, tg) for _ in range(8)]

def evaluate(S):
    net.eval(); cor = n = 0
    with torch.no_grad():
        for s1, s2, st, m, y in S:
            lg = logits(s1, s2, st, m)[:, a.lag:]
            cor += ((lg > 0).float() == y).sum().item(); n += y.numel()
    net.train(); return cor / n

def consistency_check():
    """Incremental cache vs full causal forward with same injections; max abs logit diff."""
    net.eval()
    s1, s2, st, m, y = batch(i2, N, 2, np.random.default_rng(123))
    with torch.no_grad():
        # precompute full-seq deltas once; stream uses slices; full uses all
        d_full = net.deltas(m)
        net._delta = d_full
        h_f = backbone(s1, s2, st, m)  # backbone clears _delta; recompute
        net._delta = d_full
        # re-run full logits path explicitly
        x = model.embedding([s1, s2]) + model.time_emb(st)
        for layer in model.transformer:
            x = layer(x)
        h_f = model.norm(x)
        B, T, _ = m.shape
        hs = []; pasts = [None] * len(model.transformer)
        for t in range(T):
            net._delta = {li: (d_full[li][0][:, :, t:t+1], d_full[li][1][:, :, t:t+1]) for li in sites}
            xt = model.embedding([s1[:, t:t+1], s2[:, t:t+1]]) + model.time_emb(st[:, t:t+1])
            for li, layer in enumerate(model.transformer):
                xt, pasts[li] = layer(xt, past_kv=pasts[li], use_cache=True)
            hs.append(xt)
        net._delta = {}
        h_s = model.norm(torch.cat(hs, 1))
        # note: streaming applied norm per-step incorrectly before; now norm once on cat of block outputs
        # Actually each hs[t] is block stack output at t (pre-final-norm). Final norm on full cat matches backbone.
        lg_s = net.pi(torch.cat([h_s / (h_s.norm(dim=-1, keepdim=True) + 1e-6) * 8, net.enc_head(m)], -1)).squeeze(-1)
        lg_f = net.pi(torch.cat([h_f / (h_f.norm(dim=-1, keepdim=True) + 1e-6) * 8, net.enc_head(m)], -1)).squeeze(-1)
        diff = (lg_s - lg_f).abs().max().item()
        print(json.dumps({"debug_h_maxdiff": (h_s - h_f).abs().max().item(), "debug_logit_maxdiff": diff}), flush=True)
    net.train(); return diff

ntr = sum(p.numel() for p in net.parameters())
print(json.dumps({"trainable_params": ntr, "sites": sites, "rank": a.rank, "mid": a.mid}), flush=True)

if a.cons_check:
    # quick train 0 steps, just check consistency with random init deltas
    d = consistency_check()
    print(json.dumps({"cons_max_abs": d, "pass": d < 1e-4})); sys.exit(0)

opt = torch.optim.Adam(net.parameters(), lr=1e-3); lossf = nn.BCEWithLogitsLoss()
curve = []; first60 = None; best = -1; best_step = 0; best_state = copy.deepcopy(net.state_dict()); stop_reason = "cap"
for it in range(1, a.cap + 1):
    s1, s2, st, m, y = batch(0, i1, a.B, rng)
    lg = logits(s1, s2, st, m)[:, a.lag:]
    loss = lossf(lg, y); opt.zero_grad(); loss.backward(); opt.step()
    if it % 100 == 0:
        acc = evaluate(VAL)
        curve.append({"step": it, "val_acc": acc, "loss": float(loss), "gates": net.gates()})
        if first60 is None and acc >= 0.60: first60 = it
        if acc > best + 0.001: best, best_step, best_state = acc, it, copy.deepcopy(net.state_dict())
        print(json.dumps(curve[-1]), flush=True)
        if it >= a.min_steps and it - best_step >= a.patience:
            stop_reason = "early_stop"; break

net.load_state_dict(best_state); test_acc = evaluate(TEST)
cons = consistency_check()
res = {"variant": "kv", "seed": a.seed, "lag": a.lag, "rank": a.rank, "mid": a.mid, "sites": sites,
       "trainable_params": int(ntr), "test_acc": test_acc, "best_val_acc": best, "best_step": best_step,
       "steps_run": it, "stop_reason": stop_reason, "steps_to_60_val": first60,
       "gates_at_best": net.gates(), "curve": curve, "cons_max_abs": cons,
       "n_val_targets": int(sum(v[4].numel() for v in VAL)), "n_test_targets": int(sum(v[4].numel() for v in TEST)),
       "runtime_s": round(time.time() - T0, 1)}
OUT.mkdir(parents=True, exist_ok=True)
json.dump(res, open(OUT / f"step2_kv_L{a.lag}_s{a.seed}.json", "w"), indent=1)
print("DONE", json.dumps({k: res[k] for k in ["variant","seed","lag","trainable_params","test_acc","best_step","steps_run","stop_reason","cons_max_abs","runtime_s"]}))