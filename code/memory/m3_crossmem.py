# -*- coding: utf-8 -*-
"""M3 v2 cross-memory: can a TRAINABLE read path with POSITION unlock long-lag recall
that frozen Kronos q/k geometry cannot reach?

Fix vs v1: v1's cross-attention K/V came from m only with no positional signal, so q_t
could not tell s=t-L from any other s (iid c) -> it could only average and was doomed to .5.
v2 adds RoPE over positions 0..T-1 on q and k (relative t-s), and a position-free arm as
the ablation that must fail at L>0.

Task (same as E6): m_t = [c_t iid +-1, z_t ~ N(0,1)], y_t = c_{t-L} (L=0 positive control).
Backbone: Kronos-small (8 layers, d=512, 8x64), fully frozen.

Arms:
  bypass          : per-layer gate*MLP64(m_t) on layer input (E6 multis, 663k)
  bypass_wide     : same but MLP128, param-matched to crossmem (~927k)
  crossmem_pos    : shared cross-attn, Q=<current hidden>, K/V=<m_{<=t}> + RoPE, causal, ~924k
  crossmem_nopos  : identical but NO position (must be chance for L>0)
  crossmem_slots_pos  / crossmem_slots_nopos : 32 learned slots causally resample m (16-dim
                    RoPE on slot q/k), then x_t reads slots_t (~481k)

lag=0 control: enc_head(m) is DISABLED (excluded from optimizer), so the only route to c_t
is the adapter's read path (or pointwise injection for bypass). Otherwise it is trivial.

Held-out: train [0,.70N), val [.70,.85N), test [.85N,N). VAL 8x64, TEST 16x64 in full runs.
All adapters dropout=0, SDPA scale=1/sqrt(head_dim).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(r"E:\rwkv7")
sys.path.insert(0, str(ROOT / "kronos_src"))
from model import Kronos  # noqa: E402

DATA = ROOT / "data" / "binance" / "klines_1m" / "bars_1m_of_3m.csv"
TOKENS = ROOT / "runs" / "e6_long_2026-10-20" / "e5_tokens.npz"
OUT = Path(r"C:\Users\lele\AppData\Local\Temp\opencode") / "m3_crossmem"
OUT.mkdir(parents=True, exist_ok=True)


def rms(u):
    return u / (u.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)


def mlp(w, d=512):
    return nn.Sequential(nn.Linear(2, w), nn.GELU(), nn.Linear(w, d))


def rope_tables(head_dim, n, device):
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(n, device=device, dtype=torch.float32)
    f = torch.outer(t, inv)
    emb = torch.cat((f, f), dim=-1)
    return emb.cos(), emb.sin()


def rot_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rot_half(x) * sin


def self_attn_map(layer, x):
    """Frozen backbone self-attention weights at THIS layer input (diagnostic only)."""
    B, T, _ = x.shape
    sa = layer.self_attn
    h = layer.norm1(x)
    q = sa.q_proj(h).view(B, T, sa.n_heads, sa.head_dim).transpose(1, 2)
    k = sa.k_proj(h).view(B, T, sa.n_heads, sa.head_dim).transpose(1, 2)
    q, k = sa.rotary(q, k)
    s = (q @ k.transpose(-1, -2)) / math.sqrt(sa.head_dim)
    mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()[None, None]
    return s.masked_fill(~mask, float("-inf")).softmax(-1).detach()


def addr_metrics(A, lags):
    """A: [B,H,T,T] causal attention. addr_ratio = mass(t->t-L) / mean allowed-key mass."""
    B, H, T, _ = A.shape
    rows = torch.arange(1, T + 1, device=A.device).float()
    row_mean = A.sum(-1) / rows[None, None]          # mean over allowed s (s<=t)
    out = {}
    for L in lags:
        if L >= T:
            continue
        t = torch.arange(L, T, device=A.device)
        mt = A[:, :, t, t - L].mean().item()
        base = row_mean[:, :, t].mean().item()
        peak = (A[:, :, t].max(-1).values / row_mean[:, :, t]).mean().item()
        am = (A[:, :, t].argmax(-1) == (t - L)[None, None]).float().mean().item()
        out[str(L)] = {"mass_at_target": mt, "mean_allowed": base,
                       "addr_ratio": mt / base, "addr_ratio_all": mt * T,
                       "peak_ratio": peak, "argmax_acc": am}
    return out


class CrossMem(nn.Module):
    """Shared trainable read path: x_t (query) attends to m_{<=t} (key/value), RoPE optional."""

    def __init__(self, d_model=512, n_heads=8, use_pos=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = d_model // n_heads
        self.use_pos = use_pos
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(2, d_model)
        self.v_proj = nn.Linear(2, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn = None

    def encode(self, m, rope):
        B, T, _ = m.shape
        shape = (B, T, self.n_heads, self.head_dim)
        k = self.k_proj(m).view(*shape).transpose(1, 2)  # [B,H,T,hd]
        v = self.v_proj(m).view(*shape).transpose(1, 2)
        if self.use_pos:
            cos, sin = rope
            k = apply_rope(k, cos[None, None], sin[None, None])
        return k, v

    def read(self, x, k, v, causal_mask, rope, store=False):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if self.use_pos:
            cos, sin = rope
            q = apply_rope(q, cos[None, None], sin[None, None])
        if store:
            s = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            s = s.masked_fill(~causal_mask, float("-inf"))
            self.attn = s.softmax(-1).detach()
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=causal_mask, dropout_p=0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        a = a.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(a)


class SlotMem(nn.Module):
    """32 learned slots causally resample m_{<=t} (RoPE optional), then x_t reads slots_t."""

    def __init__(self, d_model=512, n_heads=8, n_slots=32, slot_dim=64, slot_heads=4, use_pos=True):
        super().__init__()
        assert slot_dim % slot_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = d_model // n_heads
        self.n_slots, self.slot_dim, self.slot_heads = n_slots, slot_dim, slot_heads
        self.slot_hd = slot_dim // slot_heads
        self.use_pos = use_pos
        self.slots = nn.Parameter(torch.randn(n_slots, slot_dim) * 0.02)
        self.sq = nn.Linear(slot_dim, slot_dim)
        self.mk = nn.Linear(2, slot_dim)
        self.mv = nn.Linear(2, slot_dim)
        self.so = nn.Linear(slot_dim, slot_dim)
        self.xq = nn.Linear(d_model, slot_dim)
        self.sk = nn.Linear(slot_dim, slot_dim)
        self.sv = nn.Linear(slot_dim, slot_dim)
        self.xo = nn.Linear(slot_dim, d_model)

    def resample(self, m, prefix_mask, rope):
        B, T, _ = m.shape
        S, hs, hd = self.n_slots, self.slot_heads, self.slot_hd
        q0 = self.sq(self.slots)  # [S,D]
        q = q0[None, None].expand(B, T, S, self.slot_dim)
        q = q.reshape(B * T, S, hs, hd).transpose(1, 2)            # [B*T,hs,S,hd]
        k = self.mk(m).view(B, T, hs, hd).transpose(1, 2)           # [B,hs,T,hd]
        v = self.mv(m).view(B, T, hs, hd).transpose(1, 2)
        k = k.unsqueeze(1).expand(B, T, hs, T, hd).reshape(B * T, hs, T, hd)
        v = v.unsqueeze(1).expand(B, T, hs, T, hd).reshape(B * T, hs, T, hd)
        if self.use_pos:
            cos, sin = rope
            # slot query sits at time t; m key at time s -> encodes relative t-s
            ct = cos.unsqueeze(0).expand(B, T, hd).reshape(B * T, 1, 1, hd)
            st = sin.unsqueeze(0).expand(B, T, hd).reshape(B * T, 1, 1, hd)
            q = apply_rope(q, ct, st)
            k = apply_rope(k, cos[None, None, :, :], sin[None, None, :, :])
        pm = prefix_mask.view(1, T, 1, T).expand(B, T, S, T).reshape(B * T, 1, S, T)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=pm, dropout_p=0.0,
                                           scale=1.0 / math.sqrt(hd))
        a = a.transpose(1, 2).reshape(B, T, S, self.slot_dim)
        return self.so(a)

    def read(self, x, slots):
        B, T, S, D = slots.shape
        hs, hd = self.slot_heads, self.slot_hd
        q = self.xq(x).view(B, T, hs, hd).transpose(1, 2)
        k = self.sk(slots).view(B, T, S, hs, hd).permute(0, 3, 1, 2, 4).reshape(B, hs, T * S, hd)
        v = self.sv(slots).view(B, T, S, hs, hd).permute(0, 3, 1, 2, 4).reshape(B, hs, T * S, hd)
        block = torch.zeros(T, T * S, dtype=torch.bool, device=x.device)
        for t in range(T):
            block[t, t * S:(t + 1) * S] = True
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=block[None, None],
                                           dropout_p=0.0, scale=1.0 / math.sqrt(hd))
        a = a.transpose(1, 2).reshape(B, T, self.slot_dim)
        return self.xo(a)


class Net(nn.Module):
    def __init__(self, model, arm, lag, n_slots=32):
        super().__init__()
        self.model = model
        assert arm in ("bypass", "bypass_wide", "crossmem_pos", "crossmem_nopos",
                       "crossmem_slots_pos", "crossmem_slots_nopos")
        self.arm = arm
        self.use_head_m = lag > 0          # lag=0 positive control: no trivial enc_head(m)
        H = model.d_model
        NL = len(model.transformer)
        self.sites = list(range(NL))
        self.gate = nn.Parameter(torch.full((NL,), -2.0))
        self.enc_head = mlp(H)
        self.pi = nn.Sequential(nn.Linear(2 * H, 128), nn.GELU(), nn.Linear(128, 1))
        if arm == "bypass":
            self.enc = nn.ModuleList([mlp(64) for _ in self.sites])
        elif arm == "bypass_wide":
            self.enc = nn.ModuleList([mlp(128) for _ in self.sites])
        elif arm.startswith("crossmem_pos") or arm.startswith("crossmem_nopos"):
            self.cm = CrossMem(H, model.n_heads, use_pos=arm.endswith("_pos"))
        else:
            self.sm = SlotMem(H, model.n_heads, n_slots=n_slots,
                              use_pos=arm.endswith("_pos"))

    def forward(self, s1, s2, st, m, store_attn=False):
        T = m.shape[1]
        mask = torch.ones(T, T, dtype=torch.bool, device=m.device).tril()[None, None]
        rope = rope_tables(64, T, m.device) if self.arm.startswith("crossmem_pos") else None
        rope_s = rope_tables(16, T, m.device) if self.arm == "crossmem_slots_pos" else None
        x = self.model.embedding([s1, s2]) + self.model.time_emb(st)
        if self.arm in ("crossmem_pos", "crossmem_nopos"):
            kv = self.cm.encode(m, rope)
        elif self.arm in ("crossmem_slots_pos", "crossmem_slots_nopos"):
            pm = mask[0, 0][:, None, :]
            slots = self.sm.resample(m, pm, rope_s)
        self.last_attn = []
        for i, layer in enumerate(self.model.transformer):
            g = torch.sigmoid(self.gate[i])
            if self.arm == "bypass" or self.arm == "bypass_wide":
                u = rms(self.enc[i](m))
            elif self.arm in ("crossmem_pos", "crossmem_nopos"):
                u = rms(self.cm.read(x, kv[0], kv[1], mask, rope,
                                     store=store_attn and i == len(self.sites) - 1))
                if store_attn and i == len(self.sites) - 1:
                    self.last_attn.append(self.cm.attn)
            else:
                u = rms(self.sm.read(x, slots))
            x = x + g * u
            if store_attn and (self.arm == "bypass" or self.arm == "bypass_wide") \
                    and i == len(self.sites) - 1:
                self.last_attn.append(self_attn_map(layer, x))
            x = layer(x)
        h = self.model.norm(x)
        hm = self.enc_head(m) if self.use_head_m else torch.zeros_like(self.enc_head(m))
        return self.pi(torch.cat([rms(h) * 8, hm], -1)).squeeze(-1)


def bhash(model):
    h = hashlib.sha1()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def counts(net):
    out = {}
    for name, mod in net.named_children():
        if name == "model":
            continue
        out[name] = int(sum(p.numel() for p in mod.parameters()))
    out["gate"] = int(net.gate.numel())
    out["trainable_total"] = int(sum(p.numel() for p in net.parameters() if p.requires_grad))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="crossmem_pos")
    ap.add_argument("--lag", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--eval_every", type=int, default=40)
    ap.add_argument("--val_batches", type=int, default=8)
    ap.add_argument("--test_batches", type=int, default=16)
    ap.add_argument("--val_B", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--n_slots", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min_steps", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--report_params", action="store_true")
    ap.add_argument("--probe_batches", type=int, default=4)
    ap.add_argument("--probe_every", type=int, default=0)
    ap.add_argument("--tag", default="smoke")
    a = ap.parse_args()
    assert 0 <= a.lag < a.T
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    dev = a.device
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    T0 = time.time()

    df = pd.read_csv(DATA, usecols=["ts"])
    N = len(df)
    i1, i2 = int(N * 0.70), int(N * 0.85)
    z = np.load(TOKENS)
    ts = pd.to_datetime(df["ts"], unit="s", utc=True)
    STAMP = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1).astype(np.float32)
    del df, ts
    S1 = torch.tensor(z["S1"], device=dev)
    S2 = torch.tensor(z["S2"], device=dev)
    ST = torch.tensor(STAMP, device=dev)
    del z, STAMP
    print(f"data: N={N} train=[0,{i1}) val=[{i1},{i2}) test=[{i2},{N}) "
          f"T={a.T} lag={a.lag} B={a.B} dev={dev}", flush=True)

    model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    h0 = bhash(model)
    net = Net(model, a.arm, a.lag, a.n_slots).to(dev)
    pc = counts(net)
    print("params:", json.dumps(pc), flush=True)
    if a.report_params:
        print("PARAM_REPORT", json.dumps({"arm": a.arm, "lag": a.lag,
                                          "use_head_m": net.use_head_m, "counts": pc}), flush=True)
        return

    params = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    if not net.use_head_m:
        params = [(n, p) for n, p in params if not n.startswith("enc_head.")]
    bad = [n for n, _ in params if n.startswith("model.")]
    assert not bad, bad
    assert all(not p.requires_grad for p in model.parameters())
    opt = torch.optim.Adam([p for _, p in params], lr=a.lr)
    lossf = nn.BCEWithLogitsLoss()
    ntr = sum(p.numel() for _, p in params)

    def batch(lo, hi, B, g):
        T = a.T
        starts = g.integers(lo, hi - T, B)
        idx = torch.tensor(starts[:, None] + np.arange(T), device=dev)
        c = torch.tensor(g.choice([-1.0, 1.0], size=(B, T)).astype(np.float32), device=dev)
        zz = torch.tensor(g.standard_normal((B, T)).astype(np.float32), device=dev)
        y = (c[:, : T - a.lag] > 0).float()
        return S1[idx], S2[idx], ST[idx], torch.stack([c, zz], -1), y

    vg = np.random.default_rng(999)
    VAL = [batch(i1, i2, a.val_B, vg) for _ in range(a.val_batches)]
    tg = np.random.default_rng(777)
    TEST = [batch(i2, N, a.val_B, tg) for _ in range(a.test_batches)]
    print(f"val={len(VAL)}x{a.val_B} test={len(TEST)}x{a.val_B} targets/val="
          f"{sum(v[4].numel() for v in VAL)}", flush=True)

    # ---- causality: future m must not affect past logits ----
    s1, s2, st, m, y = batch(i1, i2, 2, vg)
    m2 = m.clone()
    m2[:, a.T // 2:] = torch.randn_like(m2[:, a.T // 2:])
    with torch.no_grad():
        l1 = net(s1, s2, st, m)
        l2 = net(s1, s2, st, m2)
    k = a.T // 2
    leak = (l1[:, :k] - l2[:, :k]).abs().max().item()
    causal_ok = leak < 1e-5
    print(f"causal_check: max|dlogit[:{k}]|={leak:.3e} ok={causal_ok}", flush=True)
    assert causal_ok, "causal mask leak!"

    # ---- gradient isolation ----
    lg = net(s1, s2, st, m)[:, a.lag:]
    loss = lossf(lg, y)
    opt.zero_grad(); loss.backward()
    bb_grads = [n for n, p in model.named_parameters() if p.grad is not None]
    ag = {n: float(p.grad.norm()) for n, p in params if p.grad is not None}
    assert not bb_grads, bb_grads[:5]
    opt.step()
    print(f"grad_check: backbone_grads={len(bb_grads)} adapter_tensors_with_grad={len(ag)} "
          f"max_grad={max(ag.values()):.3e} opt_params={len(params)} n={ntr}", flush=True)

    # ---- addressing probe (attention weights only, no extra training) ----
    PROBE_LAGS = [0, 10, 15, 30, 60]
    probe_batches = min(len(TEST), a.probe_batches)

    def addr_probe():
        accs = {}
        for j in range(probe_batches):
            s1_, s2_, st_, m_, y_ = TEST[j]
            with torch.no_grad():
                net(s1_, s2_, st_, m_, store_attn=True)
            if not net.last_attn:
                return None
            for L, v in addr_metrics(net.last_attn[0], PROBE_LAGS).items():
                accs.setdefault(L, []).append(v)
        n = len(accs.get("0", []))
        return {L: {k: sum(d[k] for d in vs) / n for k in vs[0]} for L, vs in accs.items()}

    probe0 = addr_probe()
    print("addr_probe_init:", json.dumps(probe0), flush=True)

    def evaluate(S):
        cor = n = 0
        with torch.no_grad():
            for s1_, s2_, st_, m_, y_ in S:
                lg_ = net(s1_, s2_, st_, m_)[:, a.lag:]
                cor += ((lg_ > 0).float() == y_).sum().item()
                n += y_.numel()
        return cor / n

    curve, best, best_step = [], -1.0, 0
    for it in range(1, a.steps + 1):
        s1, s2, st, m, y = batch(0, i1, a.B, rng)
        lg = net(s1, s2, st, m)[:, a.lag:]
        loss = lossf(lg, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % a.eval_every == 0 or it == a.steps:
            acc = evaluate(VAL)
            curve.append({"step": it, "val_acc": round(acc, 4), "loss": round(float(loss), 4)})
            if a.probe_every and it % a.probe_every == 0 and curve[-1]["step"] % a.probe_every == 0:
                curve[-1]["addr"] = addr_probe()
            if acc > best:
                best, best_step = acc, it
            print(json.dumps({"step": it, "val_acc": acc, "loss": float(loss),
                              "gates": [round(float(g), 4) for g in torch.sigmoid(net.gate)]}), flush=True)
            if a.patience and it >= a.min_steps and it - best_step >= a.patience:
                print("early_stop", flush=True)
                break

    test_acc = evaluate(TEST)
    h1 = bhash(model)
    probe1 = addr_probe()
    print("addr_probe_final:", json.dumps(probe1), flush=True)
    diag = None
    if probe1 and "0" in probe1:
        diag = {"diag_mass": probe1["0"]["mass_at_target"],
                "argmax_acc": probe1["0"]["argmax_acc"]}

    res = {
        "arm": a.arm, "lag": a.lag, "seed": a.seed, "B": a.B, "T": a.T,
        "steps": a.steps, "lr": a.lr, "device": dev,
        "use_head_m": net.use_head_m, "trainable_params": ntr, "param_counts": pc,
        "val_acc_best": best, "val_acc_best_step": best_step,
        "val_acc_final": curve[-1]["val_acc"] if curve else None,
        "test_acc": test_acc, "curve": curve,
        "causal_max_leak": leak,
        "backbone_hash_unchanged": h0 == h1,
        "backbone_grad_tensors": len(bb_grads), "n_opt_params": len(params),
        "n_val_targets": int(sum(v[4].numel() for v in VAL)),
        "n_test_targets": int(sum(v[4].numel() for v in TEST)),
        "attn_diag": diag, "addr_probe_init": probe0, "addr_probe_final": probe1,
        "runtime_s": round(time.time() - T0, 1),
    }
    js = OUT / f"m3_{a.tag}_{a.arm}_L{a.lag}_s{a.seed}.json"
    js.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("DONE", json.dumps({k: res[k] for k in
          ["arm", "lag", "trainable_params", "val_acc_best", "test_acc",
           "backbone_hash_unchanged", "runtime_s"]}), flush=True)
    print("json:", js, flush=True)


if __name__ == "__main__":
    main()