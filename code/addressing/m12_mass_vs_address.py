"""M12d+: mass vs address control for forced-attention storage probe.

CPU-only, self-contained reimplementation of the M12 design on a frozen
8-layer / 8-head / d=256 random transformer (synthetic iid-bit stream).

Arms (all share identical injection, data and probe protocol):
  force           : alpha = 1 at s* = t-L
  wrong           : alpha = 1 at a random allowed s != s*
  uniform         : all allowed logits = 0 (uniform over causal keys)
  natural         : unmodified logits
  shuffled_target : alpha = 1 at s*, but V[s*] replaced by V[r], r != s*

Probe: per-head ridge readout of c_{t-L}; head selected on val, test reported.

Usage: python m12_mass_vs_address.py
"""
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

D = 256
H = 8
DH = D // H
NL = 8
T = 96
B = 96
FF = 512
N_TRAIN, N_VAL = 48, 16
LAGS = [0, 15, 30, 60]
SITES = [("layer0", 0, 0), ("layer7", 0, 7)]  # name, inject_layer, read_layer
MODES = ["force", "wrong", "uniform", "natural", "shuffled_target"]
MAIN_AMP = 1.0
SWEEP = [0.05, 0.25, 1.00]
SWEEP_MODES = ["force", "uniform"]

DEV = "cpu"
GEN = torch.Generator(device=DEV).manual_seed(20261115)


def sinusoidal_pos(T, D):
    pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * (-math.log(10000.0) / D))
    pe = torch.zeros(T, D)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.o = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, FF)
        self.fc2 = nn.Linear(FF, D)

    def forward(self, h):
        x = self.ln1(h)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(*q.shape[:-1], H, DH).transpose(-3, -2)
        k = k.view(*k.shape[:-1], H, DH).transpose(-3, -2)
        v = v.view(*v.shape[:-1], H, DH).transpose(-3, -2)
        return q, k, v


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(NL)])

    def forward(self, h0, m, mode, amp, inj_layer, read_layer, tgt, alt):
        """Returns (head_outputs, target_mass, stats).

        head_outputs: (B,H,T,DH) from the read layer.
        tgt/alt: (B,T) int64, target and random-alternative positions.
        """
        B_, T_, _ = h0.shape
        h = h0
        feats = None
        target_mass = None
        other_mass = None
        causal = torch.tril(torch.ones(T_, T_, dtype=torch.bool, device=h.device))

        for li, blk in enumerate(self.blocks):
            if li == inj_layer:
                # amplitude aligned to the mean activation norm at the injection site
                g = amp * h.norm(dim=-1).mean()
                h = h + g * m

            q, k, v = blk(h)
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(DH)
            logits = logits.masked_fill(~causal, -1e9)

            if li == read_layer:
                if mode in ("force", "shuffled_target"):
                    mask = torch.zeros_like(logits).scatter_(
                        -1, tgt.unsqueeze(1).unsqueeze(-1).expand(B_, H, T_, 1), 1.0
                    ).bool()
                    logits = logits.masked_fill(~mask, -1e9)
                elif mode == "wrong":
                    mask = torch.zeros_like(logits).scatter_(
                        -1, alt.unsqueeze(1).unsqueeze(-1).expand(B_, H, T_, 1), 1.0
                    ).bool()
                    logits = logits.masked_fill(~mask, -1e9)
                elif mode == "uniform":
                    logits = torch.zeros_like(logits)
                    logits = logits.masked_fill(~causal, -1e9)

            attn = F.softmax(logits, dim=-1)
            out = attn @ v  # (B,H,T,DH)

            if li == read_layer:
                # diagnostics: mass on s* / on the random alternative, averaged over all queries
                aw = attn
                target_mass = aw.gather(-1, tgt.unsqueeze(1).unsqueeze(-1).expand(B_, H, T_, 1)).mean().item()
                other_mass = aw.gather(-1, alt.unsqueeze(1).unsqueeze(-1).expand(B_, H, T_, 1)).mean().item()
                if mode == "shuffled_target":
                    # keep alpha = one-hot(s*), but replace V[s*] by V[r]
                    ridx = alt.view(B_, 1, T_, 1).expand(B_, H, T_, DH)
                    out = v.gather(2, ridx)
                feats = out

            o = out.transpose(-3, -2).reshape(B_, T_, D)
            h = h + blk.o(o)
            h = h + blk.fc2(F.gelu(blk.fc1(blk.ln2(h))))

        return feats, target_mass, other_mass


def ridge_probe(Xtr, ytr, Xva, yva, Xte, yte, lam=1e-2):
    mu = Xtr.mean(0, keepdim=True)
    Xtr = Xtr - mu
    Xva = Xva - mu
    Xte = Xte - mu
    ym = ytr.mean()
    yc = ytr - ym
    A = Xtr.t() @ Xtr + lam * torch.eye(Xtr.shape[1], dtype=Xtr.dtype)
    w = torch.linalg.solve(A, Xtr.t() @ yc)
    pred_va = Xva @ w + ym
    acc_va = ((pred_va * yva) > 0).float().mean().item()
    pred_te = Xte @ w + ym
    acc_te = ((pred_te * yte) > 0).float().mean().item()
    return acc_va, acc_te


def build_data():
    pos = sinusoidal_pos(T, D).unsqueeze(0).expand(B, T, D)
    h0 = pos + 0.02 * torch.randn(B, T, D, generator=GEN)
    c = (torch.randint(0, 2, (B, T), generator=GEN, dtype=torch.float32) * 2 - 1)
    u = torch.randn(D, generator=GEN)
    u = u / u.norm()
    m = c.unsqueeze(-1) * u  # unit-norm memory vector per position
    return h0, c, m


def make_indices(c, L):
    B_, T_ = c.shape
    # t >= max(L,1): keep at least one alternative position for wrong/shuffled
    tmin = max(L, 1)
    ts = torch.arange(tmin, T_)
    tgt = ts.unsqueeze(0).expand(B_, -1) - L  # (B,Tvalid)
    hi = ts + 1  # allowed keys 0..t
    alt = (torch.rand(B_, len(ts), generator=GEN) * hi).long()
    clash = alt == tgt
    alt = torch.where(clash, (alt + 1) % hi, alt)
    return ts, tgt, alt


@torch.no_grad()
def evaluate(model, h0, c, m, L, site, mode, amp, indices=None):
    name, inj, read = site
    ts, tgt, alt = indices if indices is not None else make_indices(c, L)
    # full-length tensors for the forward; invalid queries get a safe placeholder
    tgt_full = tgt.clone()
    alt_full = alt.clone()
    # positions before tmin are not probed; force them to a valid causal index
    pad = torch.arange(0, ts[0].item())
    if len(pad):
        tgt_full = torch.cat([pad.unsqueeze(0).expand(B, -1), tgt_full], dim=1)
        alt_full = torch.cat([torch.zeros(B, len(pad), dtype=torch.long), alt_full], dim=1)
    feats, tmass, omass = model(h0, m, mode, amp, inj, read, tgt_full, alt_full)
    # feats: (B,H,T,DH); select probed positions
    feats = feats[:, :, ts, :]  # (B,H,P,DH)
    y = c.gather(1, tgt)  # (B,P)
    n = feats.shape[0]
    tr = slice(0, N_TRAIN)
    va = slice(N_TRAIN, N_TRAIN + N_VAL)
    te = slice(N_TRAIN + N_VAL, n)
    best = None
    for h in range(H):
        # split by sequence, never by position
        Xtr = feats[tr, h].reshape(-1, DH)
        ytr = y[tr].reshape(-1)
        Xva = feats[va, h].reshape(-1, DH)
        yva = y[va].reshape(-1)
        Xte = feats[te, h].reshape(-1, DH)
        yte = y[te].reshape(-1)
        acc_va, acc_te = ridge_probe(Xtr, ytr, Xva, yva, Xte, yte)
        if best is None or acc_va > best[1]:
            best = (h, acc_va, acc_te)
    return {
        "head": best[0],
        "val": round(best[1], 4),
        "test": round(best[2], 4),
        "target_mass": round(tmass, 4),
        "alt_mass": round(omass, 4),
        "n_test": int((n - (N_TRAIN + N_VAL)) * feats.shape[2]),
    }


def main():
    t0 = time.time()
    torch.manual_seed(11)
    model = TinyBackbone().eval()
    h0, c, m = build_data()
    indices = {L: make_indices(c, L) for L in LAGS}
    results = {}
    for site in SITES:
        for L in LAGS:
            for mode in MODES:
                r = evaluate(model, h0, c, m, L, site, mode, MAIN_AMP, indices[L])
                results[f"{site[0]}|L{L}|{mode}"] = r
                print(f"{site[0]:7s} L={L:2d} {mode:16s} "
                      f"head={r['head']} val={r['val']:.3f} test={r['test']:.3f} "
                      f"mass*={r['target_mass']:.4f} mass_alt={r['alt_mass']:.4f}", flush=True)

    sweep = {}
    for site in SITES:
        for L in LAGS:
            for mode in SWEEP_MODES:
                for a in SWEEP:
                    r = evaluate(model, h0, c, m, L, site, mode, a, indices[L])
                    sweep[f"{site[0]}|L{L}|{mode}|a{a}"] = r
                    print(f"SWEEP {site[0]:7s} L={L:2d} {mode:12s} a={a:.2f} "
                          f"test={r['test']:.3f}", flush=True)

    out = {"config": {"D": D, "H": H, "NL": NL, "T": T, "B": B,
                      "lags": LAGS, "sites": SITES, "main_amp": MAIN_AMP},
           "main": results, "sweep": sweep,
           "elapsed_s": round(time.time() - t0, 1)}
    with open("m12_mass_vs_address_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("DONE", out["elapsed_s"], "s")


if __name__ == "__main__":
    main()