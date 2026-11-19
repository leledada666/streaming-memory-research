# -*- coding: utf-8 -*-
"""E4: learned addressing / pointer recall.

Stream m_t = [c_t, marker_t], c_t iid +-1, marker_t in {0,1}.
One query per sequence at t_q: pick s ~ U[0, t_q-1], set marker_s=1,
target y = c_s.  Lag = t_q - s.  The model is never given the lag.

Arms (params aligned):
  fir16   : MLP on the last 16 m's (cannot see marker older than 15)
  gru128  : causal GRU-128 summary + linear head
  cross   : Q from causal GRU h_t, K/V from m_{<=t}, learned relative-pos
            bias, causal mask; no residual shortcut from m_t to head
  bypass  : 2-layer GRU, m_t injected at every layer

Conditions:
  real     : marker at true s, y=c_s
  shuffle  : one marker, position shuffled to s'!=s, y=c_s
  nomarker : marker all zero, y=c_s
  pos      : marker at t_q, y=c_{t_q}  (positive control, lag 0)

Causality: output at t only depends on positions <= t.  A numerical
future-perturbation test and an attention-mask test are printed.

CPU smoke: T=32, B=4, 200 steps, arms fir16/gru128/cross, real + controls.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(4)


# ---------------------------------------------------------------- data
def make_batch(B, T, cond, device, gen, tq_val=None, lag_range=None, balanced=True):
    """Return m [B,T,2], y [B], tq [B], s [B].  marker_s=1, y=c_s."""
    tq = torch.full((B,), (T - 1 if tq_val is None else tq_val), dtype=torch.long, device=device)
    c = torch.randint(0, 2, (B, T), device=device, generator=gen).float() * 2 - 1
    marker = torch.zeros((B, T), device=device)
    if cond == "pos":
        s = tq.clone()
        marker.scatter_(1, s[:, None], 1.0)
        y = c.gather(1, s[:, None]).squeeze(1)
    else:
        if lag_range is None and balanced:
            bs = lag_buckets(T)
            bi = torch.randint(0, len(bs), (B,), device=device, generator=gen)
            L = torch.empty(B, dtype=torch.long, device=device)
            for b in range(B):
                lo, hi = bs[int(bi[b].item())]
                L[b] = torch.randint(lo, min(hi, int(tq[0].item())) + 1, (1,),
                                     device=device, generator=gen)[0]
        else:
            lo, hi = (1, int(tq[0].item())) if lag_range is None else lag_range
            hi = min(hi, int(tq[0].item()))
            L = torch.randint(lo, hi + 1, (B,), device=device, generator=gen)
        s = tq - L
        marker.scatter_(1, s[:, None], 1.0)
        y = c.gather(1, s[:, None]).squeeze(1)
        if cond in ("shuffle", "nomarker"):
            if cond == "shuffle":
                s2 = torch.randint(0, int(tq[0].item()), (B,), device=device, generator=gen)
                for b in range(B):
                    # resample collision; tq fixed in practice
                    while int(s2[b].item()) == int(s[b].item()):
                        s2[b] = torch.randint(0, int(tq[b].item()), (1,), device=device,
                                               generator=gen)[0]
                marker.zero_()
                marker.scatter_(1, s2[:, None], 1.0)
            else:
                marker.zero_()
    m = torch.stack([c, marker], dim=-1)
    return m, y, tq, s


def lag_buckets(T):
    out = [(1, 8), (9, 16), (17, 32), (33, 64)]
    out = [(lo, min(hi, T - 1)) for lo, hi in out if lo <= T - 1]
    return out


# ---------------------------------------------------------------- models
class FIR16(nn.Module):
    def __init__(self, d=192, k=16):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(nn.Linear(2 * k, d), nn.GELU(),
                                 nn.Linear(d, d), nn.GELU(),
                                 nn.Linear(d, 1))

    def forward(self, m):
        B, T, _ = m.shape
        x = m.permute(0, 2, 1)                       # B,2,T
        x = F.pad(x, (self.k - 1, 0))                # causal left pad
        w = x.unfold(2, self.k, 1)                   # B,2,T,k
        w = w.permute(0, 2, 1, 3).reshape(B, T, 2 * self.k)
        return self.net(w).squeeze(-1)


class GRUArm(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.gru = nn.GRU(2, d, batch_first=True)
        self.head = nn.Linear(d, 1)

    def forward(self, m):
        h, _ = self.gru(m)
        return self.head(h).squeeze(-1)


class CrossAttn(nn.Module):
    def __init__(self, T=96, d=64, heads=4):
        super().__init__()
        self.d, self.h, self.T = d, heads, T
        self.emb = nn.Linear(2, d)
        self.q_gru = nn.GRU(2, d, batch_first=True)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.ln = nn.LayerNorm(d)
        self.rel = nn.Parameter(torch.zeros(heads, 2 * T - 1))
        self.head = nn.Linear(d, 1)

    def forward(self, m, return_attn=False):
        B, T, _ = m.shape
        x = self.emb(m)                              # K/V source, contains c and marker
        h, _ = self.q_gru(m)                         # Q source (causal summary)
        q = self.q(h).view(B, T, self.h, -1).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, -1).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, -1).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.d // self.h) ** 0.5
        idx = torch.arange(T, device=m.device)
        ridx = (idx[:, None] - idx[None, :]).clamp(-(T - 1), T - 1) + (T - 1)
        att = att + self.rel[:, ridx].unsqueeze(0)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=m.device), 1)
        att = att.masked_fill(causal, float("-inf")).softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(B, T, self.d)
        z = self.ln(self.ff(self.out(o)))            # deliberately no m_t residual
        logits = self.head(z).squeeze(-1)
        return (logits, att) if return_attn else logits


class BypassGRU(nn.Module):
    def __init__(self, d=80):
        super().__init__()
        self.g1 = nn.GRU(2, d, batch_first=True)
        self.g2 = nn.GRU(d + 2, d, batch_first=True)
        self.head = nn.Linear(d, 1)

    def forward(self, m):
        h1, _ = self.g1(m)
        h2, _ = self.g2(torch.cat([h1, m], dim=-1))  # inject m_t at every layer
        return self.head(h2).squeeze(-1)


def build(arm, T, device):
    if arm == "fir16":
        return FIR16(192)
    if arm == "gru128":
        return GRUArm(128)
    if arm == "cross":
        return CrossAttn(T=T, d=64, heads=4)
    if arm == "bypass":
        return BypassGRU(80)
    raise ValueError(arm)


def nparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- train / eval
def train(arm, cond, T, B, steps, seed, device, lr=3e-3, log_every=50, balanced=True):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed + 777)
    model = build(arm, T, device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ar = torch.arange(B, device=device)
    t0 = time.time()
    for step in range(1, steps + 1):
        m, y, tq, s = make_batch(B, T, cond, device, gen, balanced=balanced)
        logits = model(m)
        loss = F.binary_cross_entropy_with_logits(logits[ar, tq], (y > 0).float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and (step % log_every == 0 or step == 1):
            with torch.no_grad():
                pred = torch.where(logits[ar, tq] > 0, 1.0, -1.0)
                acc = (pred == y).float().mean().item()
            print(json.dumps({"arm": arm, "cond": cond, "step": step,
                              "loss": round(loss.item(), 4), "batch_acc": round(acc, 3),
                              "elapsed_s": round(time.time() - t0, 1)}), flush=True)
    return model


@torch.no_grad()
def eval_model(model, cond, T, device, per_bucket=64, seed=1234, attn=False):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    rows, attn_mass, counts = {}, {}, {}
    sizes = [0] if cond == "pos" else lag_buckets(T)
    for lo, hi in ([(0, 0)] if cond == "pos" else lag_buckets(T)):
        B = per_bucket
        m, y, tq, s = make_batch(B, T, cond, device, gen, lag_range=(lo, hi))
        out = model(m, return_attn=True) if attn else model(m)
        if attn:
            logits, A = out
            # mass on the true marker position s at the query tq
            # per-sample loop: safer than advanced indexing on CUDA
            mass = 0.0
            for b in range(B):
                mass += A[b, :, tq[b].item(), s[b].item()].sum().item()
            attn_mass[f"{lo}-{hi}"] = round(mass / B, 3)
        else:
            logits = out
        pred = torch.where(logits[:, tq[0]] > 0, 1.0, -1.0)
        acc = (pred == y).float().mean().item()
        key = f"{lo}-{hi}"
        rows[key] = round(acc, 4)
        counts[key] = B
    # overall on a natural uniform-lag batch
    B = per_bucket * 2
    m, y, tq, s = make_batch(B, T, cond, device, gen, lag_range=(1, T - 1))
    logits = model(m)
    pred = torch.where(logits[:, tq[0]] > 0, 1.0, -1.0)
    overall = (pred == y).float().mean().item()
    model.train()
    return {"buckets": rows, "counts": counts, "overall": round(overall, 4),
            "attn_mass_at_s": attn_mass if attn else None}


# ---------------------------------------------------------------- causality
@torch.no_grad()
def causal_check(model, name, T=32, device="cpu", seed=99):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    m, y, tq, s = make_batch(2, T, "real", device, gen, tq_val=12, lag_range=(1, 12))
    l1 = model(m)
    if isinstance(l1, tuple):
        l1 = l1[0]
    m2 = m.clone()
    # randomize every position strictly after t=12
    m2[:, 13:, 0] = torch.randint(0, 2, (2, T - 13), device=device, generator=gen).float() * 2 - 1
    m2[:, 13:, 1] = torch.randint(0, 2, (2, T - 13), device=device, generator=gen).float()
    l2 = model(m2)
    if isinstance(l2, tuple):
        l2 = l2[0]
    diff = (l1[:, 12] - l2[:, 12]).abs().max().item()
    attn_ok = None
    if name == "cross":
        model.train(False)
        _, A = model(m, return_attn=True)
        # A: [B,H,T,T]; future j>t mass
        fut = A[:, :, 12, 13:].abs().sum().item()
        attn_ok = (fut < 1e-6)
    print(json.dumps({
        "causal_check": name, "T": T, "tq": 12,
        "s": s.tolist(), "lag": (tq - s).tolist(), "y": y.tolist(),
        "visible_positions": list(range(0, 13)),
        "future_positions": list(range(13, T)),
        "future_perturb_max_logit_diff": diff,
        "attention_future_mass_zero": attn_ok,
        "pass": bool(diff < 1e-5 and (attn_ok is not False)),
    }), flush=True)
    model.train()
    return diff


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--arms", default="fir16,gru128,cross")
    ap.add_argument("--conds", default="real")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval_per_bucket", type=int, default=64)
    ap.add_argument("--balanced", type=int, default=1)
    ap.add_argument("--out", default=str(Path(__file__).parent / "e4_smoke"))
    ap.add_argument("--tag", default="smoke")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    conds = [x.strip() for x in a.conds.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",")]
    dev = a.device

    rep = {"config": vars(a), "params": {}, "runs": []}
    for arm in arms:
        rep["params"][arm] = nparams(build(arm, a.T, dev))
    print(json.dumps({"params": rep["params"]}), flush=True)

    # causality check on untrained models (mask/architecture property)
    for arm in arms:
        causal_check(build(arm, a.T, dev).to(dev), arm, T=a.T, device=dev)

    t0 = time.time()
    for arm in arms:
        for cond in conds:
            for seed in seeds:
                model = train(arm, cond, a.T, a.B, a.steps, seed, dev, lr=a.lr,
                              balanced=bool(a.balanced))
                ev = eval_model(model, cond, a.T, dev,
                                per_bucket=a.eval_per_bucket, seed=seed + 5000,
                                attn=(arm == "cross"))
                # zero-shot controls for a model trained on real
                zs = {}
                if cond == "real":
                    for c2 in ("shuffle", "nomarker"):
                        zs[c2] = eval_model(model, c2, a.T, dev,
                                            per_bucket=a.eval_per_bucket,
                                            seed=seed + 6000, attn=(arm == "cross"))
                rec = {"arm": arm, "cond": cond, "seed": seed,
                       "params": rep["params"][arm], "eval": ev, "zero_shot": zs}
                rep["runs"].append(rec)
                print(json.dumps({"result": rec}), flush=True)

    rep["runtime_s"] = round(time.time() - t0, 1)
    with open(out / f"e4_{a.tag}.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)

    # compact table
    print("\n=== E4 SMOKE TABLE (acc by marker lag) ===")
    hdr = "arm/cond".ljust(22) + "".join(f"{k:>10}" for k in
          ([f"{lo}-{hi}" for lo, hi in lag_buckets(a.T)] + ["overall"]))
    print(hdr)
    for r in rep["runs"]:
        e = r["eval"]
        keys = list(e["buckets"].keys())
        line = f'{r["arm"]}/{r["cond"]}'.ljust(22)
        line += "".join(f'{e["buckets"][k]:>10.3f}' for k in keys)
        line += f'{e["overall"]:>10.3f}'
        print(line)
        if r["zero_shot"]:
            for c2, z in r["zero_shot"].items():
                line = f'  {r["arm"]}/{c2}(zs)'.ljust(22)
                line += "".join(f'{z["buckets"][k]:>10.3f}' for k in keys)
                line += f'{z["overall"]:>10.3f}'
                print(line)
    print("DONE", rep["runtime_s"], flush=True)


if __name__ == "__main__":
    main()