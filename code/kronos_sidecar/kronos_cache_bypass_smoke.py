# -*- coding: utf-8 -*-
"""Kronos KV-cache + gated residual bypass smoke test.

Installs nothing outside E:\\rwkv7. Freeze Kronos-small backbone.
1) Numerically verify incremental KV cache == full-window forward.
2) Train a tiny next-step inject probe / short planted-edge RL to show bypass works.
No orders, no Gate writes, no Price bot.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(r"E:\rwkv7")
SRC = ROOT / "kronos_src"
MODEL_DIR = ROOT / "models" / "Kronos-small"
TOK_DIR = ROOT / "models" / "Kronos-Tokenizer-base"
OUT = ROOT / "runs" / "kronos_cache_bypass_smoke"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC))
from model import Kronos  # patched with forward_hidden + KV cache

torch.manual_seed(0)
np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()

print("Loading Kronos-small from", MODEL_DIR)
model = Kronos.from_pretrained(str(MODEL_DIR))
model.to(DEV).eval()
for p in model.parameters():
    p.requires_grad = False
H = model.d_model
print(f"d_model={H} n_layers={model.n_layers} n_heads={model.n_heads} device={DEV}")


def make_stamp(B, T, start_minute=0):
    # [minute, hour, weekday, day, month]
    stamp = torch.zeros(B, T, 5, device=DEV)
    for t in range(T):
        m = (start_minute + t) % 60
        h = ((start_minute + t) // 60) % 24
        stamp[:, t, 0] = m
        stamp[:, t, 1] = h
        stamp[:, t, 2] = 0
        stamp[:, t, 3] = 1
        stamp[:, t, 4] = 1
    return stamp


# ---------- 1) KV cache numerical check ----------
print("\n=== KV cache check ===")
B, L = 2, 48
s1 = torch.randint(0, model.s1_vocab_size, (B, L), device=DEV)
s2 = torch.randint(0, 2 ** model.s2_bits, (B, L), device=DEV)
stamp = make_stamp(B, L)

with torch.no_grad():
    h_full = model.forward_hidden(s1, s2, stamp=stamp, use_cache=False)
    past = None
    chunks = []
    for t in range(L):
        h_t, past = model.forward_hidden(
            s1[:, t : t + 1],
            s2[:, t : t + 1],
            stamp=stamp[:, t : t + 1],
            past_key_values=past,
            use_cache=True,
        )
        chunks.append(h_t)
    h_inc = torch.cat(chunks, dim=1)

diff = (h_full - h_inc).abs()
max_abs = float(diff.max())
mean_abs = float(diff.mean())
print(f"full vs incremental: max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}")
# Also check prime-then-step: first half full with cache, then step
with torch.no_grad():
    mid = L // 2
    h_prime, past2 = model.forward_hidden(
        s1[:, :mid], s2[:, :mid], stamp=stamp[:, :mid], use_cache=True
    )
    chunks2 = [h_prime]
    for t in range(mid, L):
        h_t, past2 = model.forward_hidden(
            s1[:, t : t + 1],
            s2[:, t : t + 1],
            stamp=stamp[:, t : t + 1],
            past_key_values=past2,
            use_cache=True,
        )
        chunks2.append(h_t)
    h_mixed = torch.cat(chunks2, dim=1)
diff2 = (h_full - h_mixed).abs()
max_abs_mixed = float(diff2.max())
print(f"prime+step vs full: max_abs={max_abs_mixed:.6e}")

cache_ok = max_abs < 1e-4 and max_abs_mixed < 1e-4
print("CACHE_OK" if cache_ok else "CACHE_FAIL")


# ---------- 2) Gated residual bypass + heads ----------
class BypassNet(nn.Module):
    def __init__(self, d_model, m_dim=4, n_actions=4):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(m_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.gate_logit = nn.Parameter(torch.tensor(0.0))
        self.pi = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, n_actions))
        self.v = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1))
        # next-bar probe: predict scalar return from h_t
        self.probe = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))

    def delta(self, m):
        # m: [B, T, m_dim] -> [B, T, d_model]
        return torch.sigmoid(self.gate_logit) * self.enc(m)

    def heads(self, h):
        z = h / (h.norm(dim=-1, keepdim=True) + 1e-6) * 8
        return self.pi(z), self.v(z).squeeze(-1)

    def probe_out(self, h):
        return self.probe(h).squeeze(-1)


net = BypassNet(H).to(DEV)
opt = torch.optim.Adam(net.parameters(), lr=3e-4)

# Synthetic planted edge (same spirit as RWKV experiment)
LAG, Ksig, PHI, MAXH, HOLD_PEN = 3, 0.5, 0.9, 20, 0.002
N = 8000
rng = np.random.default_rng(0)
s = np.zeros(N, np.float32)
e = rng.standard_normal(N).astype(np.float32)
for t in range(1, N):
    s[t] = PHI * s[t - 1] + math.sqrt(1 - PHI * PHI) * e[t]
r = np.zeros(N, np.float32)
r[LAG:] = Ksig * s[:-LAG]
r += rng.standard_normal(N).astype(np.float32) * 0.1

# Dummy token stream (backbone frozen; signal via bypass)
# Use constant mid tokens so embedding is stable "carrier"
s1_const = model.s1_vocab_size // 2
s2_const = (2 ** model.s2_bits) // 2


print("\n=== Next-bar inject probe (bypass) ===")
# Predict r_{t+1} from injected m_t = [r_t, s_t, 0, 0] via h_t
T_probe = 64
B_probe = 8
probe_steps = 40
probe_log = []
for step in range(probe_steps):
    starts = rng.integers(0, N - T_probe - 2, size=B_probe)
    idx = starts[:, None] + np.arange(T_probe)[None, :]
    R = torch.tensor(r[idx], device=DEV)
    S = torch.tensor(s[idx], device=DEV)
    target = torch.tensor(r[idx + 1], device=DEV)  # next-bar return
    m = torch.stack([R * 10, S, torch.zeros_like(R), torch.zeros_like(R)], -1)
    s1 = torch.full((B_probe, T_probe), s1_const, device=DEV, dtype=torch.long)
    s2 = torch.full((B_probe, T_probe), s2_const, device=DEV, dtype=torch.long)
    stamp = make_stamp(B_probe, T_probe)

    # With bypass
    delta = net.delta(m)
    h = model.forward_hidden(s1, s2, stamp=stamp, bypass_delta=delta, use_cache=False)
    pred = net.probe_out(h)
    loss = F.mse_loss(pred, target)

    # Baseline: no useful inject (zeros) — only for logging every 10 steps
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()

    if step % 10 == 0 or step == probe_steps - 1:
        with torch.no_grad():
            h0 = model.forward_hidden(
                s1, s2, stamp=stamp,
                bypass_delta=torch.zeros_like(delta),
                use_cache=False,
            )
            # fresh probe on frozen zero-bypass path using same probe head weights
            pred0 = net.probe_out(h0)
            mse0 = float(F.mse_loss(pred0, target))
            mse1 = float(loss)
            gate = float(torch.sigmoid(net.gate_logit))
        probe_log.append({"step": step, "mse_bypass": mse1, "mse_zero": mse0, "gate": gate})
        print(f"probe step {step}: mse_bypass={mse1:.4f} mse_zero={mse0:.4f} gate={gate:.3f}")


print("\n=== Short planted-edge RL smoke (cache step mode) ===")
# Minimal PPO-lite few iters; actions: 0 hold, 1 long, 2 short, 3 close
def legal(pos):
    msk = torch.zeros(pos.shape[0], 4, dtype=torch.bool, device=DEV)
    msk[:, 0] = True
    flat = pos == 0
    msk[flat, 1] = True
    msk[flat, 2] = True
    msk[~flat, 3] = True
    return msk


@torch.no_grad()
def rollout(starts, T, greedy=False):
    B = len(starts)
    idx = torch.tensor(starts, device=DEV)[:, None] + torch.arange(T, device=DEV)
    R = torch.tensor(r, device=DEV)[idx]
    S = torch.tensor(s, device=DEV)[idx]
    pos = torch.zeros(B, device=DEV)
    held = torch.zeros(B, device=DEV)
    acc = torch.zeros(B, device=DEV)
    M, A, LP, RW, LM = [], [], [], [], []
    trades = []
    past = None
    s1_row = torch.full((B, 1), s1_const, device=DEV, dtype=torch.long)
    s2_row = torch.full((B, 1), s2_const, device=DEV, dtype=torch.long)
    for t in range(T):
        m = torch.stack([R[:, t] * 10, S[:, t], pos, held / MAXH], -1)
        lm = legal(pos)
        stamp_t = make_stamp(B, 1, start_minute=t)
        delta = net.delta(m[:, None])
        h, past = model.forward_hidden(
            s1_row, s2_row, stamp=stamp_t, past_key_values=past, use_cache=True, bypass_delta=delta
        )
        logits, _ = net.heads(h[:, -1])
        logits = logits.masked_fill(~lm, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        act = logits.argmax(-1) if greedy else dist.sample()
        lp = dist.log_prob(act)
        rew = torch.zeros(B, device=DEV)
        force = (pos != 0) & (held >= MAXH)
        if t == T - 1:
            force = force | (pos != 0)
        closing = ((act == 3) | force) & (pos != 0)
        for b in torch.nonzero(closing).flatten().tolist():
            trades.append((float(acc[b]), float(held[b])))
        rew[closing] = acc[closing] - HOLD_PEN * held[closing]
        pos = torch.where(closing, torch.zeros_like(pos), pos)
        held[closing] = 0
        acc[closing] = 0
        opening = (pos == 0) & ~closing & ((act == 1) | (act == 2))
        pos = torch.where(opening & (act == 1), torch.ones_like(pos), pos)
        pos = torch.where(opening & (act == 2), -torch.ones_like(pos), pos)
        if t + 1 < T:
            acc = acc + pos * R[:, t + 1]
            held = held + (pos != 0).float()
        M.append(m)
        A.append(act)
        LP.append(lp)
        RW.append(rew)
        LM.append(lm)
    st = lambda xs: torch.stack(xs, 1)
    return st(M), st(A), st(LP), st(RW), st(LM), trades


def ppo_update(M, A, LP, RW, LM):
    ret = torch.flip(torch.cumsum(torch.flip(RW, [1]), 1), [1])
    T = M.shape[1]
    B = M.shape[0]
    s1 = torch.full((B, T), s1_const, device=DEV, dtype=torch.long)
    s2 = torch.full((B, T), s2_const, device=DEV, dtype=torch.long)
    stamp = make_stamp(B, T)
    for _ in range(2):
        delta = net.delta(M)
        h = model.forward_hidden(s1, s2, stamp=stamp, bypass_delta=delta, use_cache=False)
        logits, v = net.heads(h)
        logits = logits.masked_fill(~LM, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        lp = dist.log_prob(A)
        adv = (ret - v).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        ratio = torch.exp(lp - LP)
        pl = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
        loss = pl + 0.5 * ((v - ret) ** 2).mean() - 0.01 * dist.entropy().mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return float(pl)


rl_iters = 12
rl_T = 64
rl_B = 8
rl_log = []
for it in range(rl_iters):
    starts = rng.integers(0, N - rl_T - 2, size=rl_B).tolist()
    M, A, LP, RW, LM, _ = rollout(starts, rl_T, greedy=False)
    # detach LP for PPO ratio base
    pl = ppo_update(M, A, LP.detach(), RW, LM)
    if it % 4 == 0 or it == rl_iters - 1:
        # quick eval
        ev_starts = list(range(6000, 7000 - rl_T, rl_T))[:16]
        *_, trades = rollout(ev_starts, rl_T, greedy=True)
        pn = np.array([x[0] for x in trades]) if trades else np.zeros(0)
        gate = float(torch.sigmoid(net.gate_logit))
        row = {
            "iter": it,
            "pl": pl,
            "n_trades": int(len(trades)),
            "total_pnl": float(pn.sum()) if len(pn) else 0.0,
            "avg_pnl": float(pn.mean()) if len(pn) else 0.0,
            "gate": gate,
        }
        rl_log.append(row)
        print(f"rl iter {it}: trades={row['n_trades']} total_pnl={row['total_pnl']:.3f} avg={row['avg_pnl']:.4f} gate={gate:.3f}")

# Save
ckpt = {
    "bypass": net.state_dict(),
    "cache_max_abs": max_abs,
    "cache_max_abs_mixed": max_abs_mixed,
    "cache_ok": cache_ok,
    "probe_log": probe_log,
    "rl_log": rl_log,
    "model": "NeoQuasar/Kronos-small",
    "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
}
torch.save({k: v for k, v in ckpt.items() if k == "bypass"}, OUT / "bypass_heads.pt")
summary = {k: v for k, v in ckpt.items() if k != "bypass"}
summary["runtime_s"] = round(time.time() - t0, 1)
summary["gate_final"] = float(torch.sigmoid(net.gate_logit))
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\nWrote", OUT / "summary.json")
print("RUNTIME_S", summary["runtime_s"])
print("DONE cache_ok=", cache_ok, "gate=", summary["gate_final"])
