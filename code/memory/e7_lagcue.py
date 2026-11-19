# -*- coding: utf-8 -*-
"""E7: capacity + lag-cue generalization (frozen Kronos-small, no RL, no orders).

Stream: m_t = [c_t^1..c_t^B, z_t], c iid +-1, z ~ N(0,1).
Query: at time t a lag cue l (discrete token) is given; target y_t = c_{t-l}
(all B bits).  Train lags {5,10}; test unseen {15,30,60}; T=32 smoke / 96 full.
Arms:
  crossattn: Q=f([h_t,cue]), K/V=past m, relative-position kernel centred on cue l.
  gru:       [m_t,cue] into GRUCell of state S in {16,64,256}; head sees [state,h_t].
  bypass:    cue concatenated to current m, gated injection at every frozen layer
             input (width 64), head sees [h,enc(m_t,cue)]; bounded by state rank R.
Controls (evaluation of the same weights): random cue, shuffled m, cue=0 (no cue);
plus L=0 positive control as a separate task (target c_t) and capacity sweep B/S.
Cue encoding: fixed log-Fourier code of l (defined for unseen l, unlike a lookup
embedding whose unseen rows never train).  Tokens/STAMP are the E6 ones.

Read-only use of E:\rwkv7.  Outputs go to --out (default under %TEMP%).
Causal self-check: perturb m/cue at t>t0, assert logits[:, :t0+1] unchanged.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(r"E:\rwkv7")
sys.path.insert(0, str(ROOT / "kronos_src"))
from model import Kronos  # noqa: E402

DATA = ROOT / "data" / "binance" / "klines_1m" / "bars_1m_of_3m.csv"
TOK = ROOT / "runs" / "e6_long_2026-10-20" / "e5_tokens.npz"
DEF_OUT = Path(r"C:\Users\lele\AppData\Local\Temp\opencode\e7_lagcue_smoke")
MAXLAG = 64
CUE_DIM = 32
H = 512


def cue_feat(lag: torch.Tensor) -> torch.Tensor:
    """lag [B,T] int -> [B,T,9] fixed smooth code; defined for all 0..MAXLAG."""
    x = lag.float().clamp(0, MAXLAG)
    l = torch.log1p(x)
    fr = torch.tensor([1.0, 2.0, 4.0, 8.0], device=x.device) * math.pi / math.log(MAXLAG + 1)
    z = l[..., None] * fr
    return torch.cat([torch.sin(z), torch.cos(z), (x / MAXLAG)[..., None]], -1)


def mlp(i, w):
    return nn.Sequential(nn.Linear(i, w), nn.GELU(), nn.Linear(w, H))


class CrossAttn(nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.bits = bits
        self.hh, self.dh = 4, 64
        self.cue = nn.Sequential(nn.Linear(9, CUE_DIM), nn.GELU())
        self.q = nn.Linear(H + CUE_DIM, self.hh * self.dh)
        self.k = nn.Linear(bits + 1, self.hh * self.dh)
        self.v = nn.Linear(bits + 1, self.hh * self.dh)
        self.o = nn.Linear(self.hh * self.dh, H)
        self.a = nn.Parameter(torch.full((self.hh,), 6.0))   # kernel width (sharp-ish init)
        self.beta = nn.Parameter(torch.zeros(self.hh))       # linear skew
        self.head = nn.Sequential(nn.Linear(2 * H, 256), nn.GELU(), nn.Linear(256, bits))

    def forward(self, h, m, lag):
        B, T, _ = m.shape
        q = self.q(torch.cat([h, self.cue(cue_feat(lag))], -1)).view(B, T, self.hh, self.dh)
        k = self.k(m).view(B, T, self.hh, self.dh)
        v = self.v(m).view(B, T, self.hh, self.dh)
        d = (torch.arange(T, device=m.device)[:, None] - torch.arange(T, device=m.device)[None, :]).float()
        delta = (d[None, None] - lag.float()[:, None, None]) / 4.0
        bias = (-F.softplus(self.a)[None, :, None, None] * delta ** 2
                + self.beta[None, :, None, None] * delta)
        att = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.dh)
        att = att + bias
        mask = torch.arange(T, device=m.device)[None, :] <= torch.arange(T, device=m.device)[:, None]
        att = att.masked_fill(~mask[None, None], -1e30)
        p = torch.softmax(att, -1)
        out = torch.einsum("bhts,bshd->bthd", p, v).reshape(B, T, -1)
        return self.head(torch.cat([h, self.o(out)], -1))


class GRUArm(nn.Module):
    def __init__(self, bits, S):
        super().__init__()
        self.bits, self.S = bits, S
        self.cue = nn.Sequential(nn.Linear(9, CUE_DIM), nn.GELU())
        self.inp = nn.Linear(bits + 1 + CUE_DIM, S)
        self.cell = nn.GRUCell(S, S)
        self.head = nn.Sequential(nn.Linear(S + H, 128), nn.GELU(), nn.Linear(128, bits))

    def forward(self, h, m, lag):
        B, T, _ = m.shape
        x = torch.cat([m, self.cue(cue_feat(lag))], -1)
        g = m.new_zeros(B, self.S)
        outs = []
        for t in range(T):
            g = self.cell(self.inp(x[:, t]), g)
            outs.append(g)
        gs = torch.stack(outs, 1)
        return self.head(torch.cat([h, gs], -1))


class Bypass(nn.Module):
    def __init__(self, bits, model, width=64):
        super().__init__()
        self.bits, self.model = bits, model
        nl = len(model.transformer)
        self.enc = nn.ModuleList([mlp(bits + 1 + CUE_DIM, width) for _ in range(nl)])
        self.g = nn.Parameter(torch.full((nl,), -2.0))
        self.cue = nn.Sequential(nn.Linear(9, CUE_DIM), nn.GELU())
        self.enc_head = mlp(bits + 1 + CUE_DIM, H)
        self.pi = nn.Sequential(nn.Linear(2 * H, 128), nn.GELU(), nn.Linear(128, bits))

    def forward(self, s1, s2, st, m, lag):
        x = self.model.embedding([s1, s2]) + self.model.time_emb(st)
        u = torch.cat([m, self.cue(cue_feat(lag))], -1)
        for i, layer in enumerate(self.model.transformer):
            e = self.enc[i](u)
            e = e / (e.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)
            x = x + torch.sigmoid(self.g[i]) * e
            x = layer(x)
        h = self.model.norm(x)
        hm = self.enc_head(u)
        return self.pi(torch.cat([h / (h.norm(dim=-1, keepdim=True) + 1e-6) * 8, hm], -1))


def build(variant, bits, S, model):
    if variant == "crossattn":
        return CrossAttn(bits)
    if variant == "gru":
        return GRUArm(bits, S)
    if variant == "bypass":
        return Bypass(bits, model)
    raise ValueError(variant)


def forward(net, variant, model, s1, s2, st, m, lag):
    if variant == "bypass":
        return net(s1, s2, st, m, lag)
    with torch.no_grad():
        x = model.embedding([s1, s2]) + model.time_emb(st)
        for layer in model.transformer:
            x = layer(x)
        h = model.norm(x)
    return net(h, m, lag)


class Data:
    def __init__(self, T, dev):
        df = pd.read_csv(DATA, usecols=["ts"])
        self.N = len(df)
        ts = pd.to_datetime(df["ts"], unit="s", utc=True)
        stamp = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1)
        z = np.load(TOK)
        self.s1 = torch.tensor(z["S1"], dtype=torch.long, device=dev)
        self.s2 = torch.tensor(z["S2"], dtype=torch.long, device=dev)
        self.st = torch.tensor(stamp.astype(np.float32), device=dev)
        self.i1, self.i2 = int(self.N * 0.70), int(self.N * 0.85)
        self.T, self.dev = T, dev

    def batch(self, lo, hi, B, lags, rng, control="none", step_lag=None):
        st0 = rng.integers(lo, hi - self.T, B)
        idx = torch.tensor(st0[:, None] + np.arange(self.T)[None, :], device=self.dev)
        s1, s2, st = self.s1[idx], self.s2[idx], self.st[idx]
        c = torch.tensor(rng.choice([-1.0, 1.0], size=(B, self.T, self.bits)).astype(np.float32), device=self.dev)
        z = torch.tensor(rng.standard_normal((B, self.T, 1)).astype(np.float32), device=self.dev)
        m = torch.cat([c, z], -1)
        lag_row = torch.tensor(np.asarray(rng.choice(lags, B)), dtype=torch.long, device=self.dev)
        tdiff = torch.arange(self.T, device=self.dev)[None, :] - lag_row[:, None]
        valid = tdiff >= 0
        src = tdiff.clamp(min=0)
        y = c[torch.arange(B, device=self.dev)[:, None], src]
        lag_t = lag_row[:, None].expand(B, self.T).clone()
        if control == "randcue":
            lag_t = torch.tensor(rng.integers(1, MAXLAG + 1, size=(B, 1)), device=self.dev).expand(B, self.T).clone()
        elif control == "nocue":
            lag_t = torch.zeros(B, self.T, dtype=torch.long, device=self.dev)
        if control == "shuffle":
            perm = torch.tensor(np.stack([rng.permutation(self.T) for _ in range(B)]), device=self.dev)
            m = m[torch.arange(B, device=self.dev)[:, None], perm]
        return s1, s2, st, m, lag_t, y, valid, c


def masked_acc(logits, y, valid):
    pred = (logits > 0).float() == (y > 0).float()
    return float(pred[valid].float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="crossattn")
    ap.add_argument("--S", type=int, default=64, help="GRU state size")
    ap.add_argument("--bits", type=int, default=1, help="B independent bits per step")
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_batches", type=int, default=8)
    ap.add_argument("--eval_B", type=int, default=32)
    ap.add_argument("--train_lags", default="5,10")
    ap.add_argument("--test_lags", default="5,10,30")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(DEF_OUT))
    ap.add_argument("--tag", default="")
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()

    if a.summarize:
        summarize(Path(a.out))
        return

    dev = a.device
    if dev == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    try:
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    except Exception:
        pass

    train_lags = [int(x) for x in a.train_lags.split(",") if x != ""]
    test_lags = [int(x) for x in a.test_lags.split(",") if x != ""]
    for L in train_lags + test_lags:
        if not (0 <= L < a.T):
            raise SystemExit(f"lag {L} invalid for T={a.T}")

    model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    data = Data(a.T, dev)
    data.bits = a.bits
    net = build(a.variant, a.bits, a.S, model).to(dev)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=a.lr)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    nparam = int(sum(p.numel() for p in net.parameters() if p.requires_grad))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"e7{a.tag}_{a.variant}"
    if a.variant == "gru":
        name += f"_S{a.S}"
    name += f"_{a.bits}bit_T{a.T}_s{a.seed}"
    t0 = time.time()

    vg = np.random.default_rng(10_000 + a.seed)
    val = [data.batch(data.i1, data.i2, a.eval_B, train_lags, vg, "none") for _ in range(4)]

    def val_acc():
        net.eval()
        c = n = 0.0
        with torch.no_grad():
            for s1, s2, st, m, lag, y, valid, _ in val:
                c += masked_acc(forward(net, a.variant, model, s1, s2, st, m, lag), y, valid)
                n += 1
        net.train()
        return c / n

    curve, best, best_step, best_state = [], -1.0, 0, None
    for it in range(1, a.steps + 1):
        s1, s2, st, m, lag, y, valid, _ = data.batch(0, data.i1, a.batch, train_lags, vg, "none")
        logits = forward(net, a.variant, model, s1, s2, st, m, lag)
        loss = (lossf(logits, (y > 0).float()) * valid[..., None]).sum() / valid.sum() / a.bits
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % a.eval_every == 0 or it == a.steps:
            va = val_acc()
            curve.append({"step": it, "val_acc": va, "loss": float(loss)})
            if va > best:
                best, best_step, best_state = va, it, {k: v.detach().clone() for k, v in net.state_dict().items()}
            print(json.dumps(curve[-1]), flush=True)
    if best_state is not None:
        net.load_state_dict(best_state)

    # fixed eval sets per (lag, condition)
    def eval_one(L, cond):
        rng = np.random.default_rng(20_000 + 97 * L + a.seed)
        c = n = 0.0
        net.eval()
        with torch.no_grad():
            for _ in range(a.eval_batches):
                s1, s2, st, m, lag, y, valid, _ = data.batch(
                    data.i2, data.N, a.eval_B, [L] * a.eval_B, rng, cond)
                lg = forward(net, a.variant, model, s1, s2, st, m, lag)
                c += masked_acc(lg, y, valid)
                n += 1
        net.train()
        return c / n

    accs = {}
    for L in test_lags:
        accs[str(L)] = eval_one(L, "none")
    ctrl = {}
    for cond in ("randcue", "shuffle", "nocue"):
        ctrl[cond] = {str(L): eval_one(L, cond) for L in test_lags}

    # causal self-check
    net.eval()
    cc = data.batch(data.i2, data.N, min(a.eval_B, 8), [max(train_lags)] * 8,
                    np.random.default_rng(7), "none")
    s1, s2, st, m, lag, y, valid, c = cc
    with torch.no_grad():
        lg1 = forward(net, a.variant, model, s1, s2, st, m, lag)
        t0c = a.T // 2
        m2, lag2 = m.clone(), lag.clone()
        m2[:, t0c + 1:] = torch.randn_like(m2[:, t0c + 1:])
        lag2[:, t0c + 1:] = torch.randint(1, MAXLAG + 1, lag2[:, t0c + 1:].shape, device=dev)
        lg2 = forward(net, a.variant, model, s1, s2, st, m2, lag2)
        causal = float((lg1[:, :t0c + 1] - lg2[:, :t0c + 1]).abs().max())
        future_changed = float((lg1[:, t0c + 1:] - lg2[:, t0c + 1:]).abs().max())
    # target audit: y_t == c_{t-L}
    src = torch.arange(a.T, device=dev)[None, :] - lag[:, :1]
    vv = src >= 0
    audit = float((y[vv] - c[torch.arange(8, device=dev)[:, None].expand_as(src)[vv], src.clamp(min=0)[vv]]).abs().max())

    res = {
        "variant": a.variant, "S": a.S if a.variant == "gru" else None, "bits": a.bits,
        "T": a.T, "batch": a.batch, "steps": a.steps, "seed": a.seed, "device": dev,
        "train_lags": train_lags, "test_lags": test_lags, "trainable_params": nparam,
        "best_val_acc": best, "best_step": best_step,
        "acc": accs, "controls": ctrl,
        "causal_max_abs": causal, "future_sensitivity": future_changed,
        "target_audit_max_abs": audit, "runtime_s": round(time.time() - t0, 1),
        "curve": curve,
    }
    json.dump(res, open(out / f"{name}.json", "w"), indent=1)
    print("RESULT " + json.dumps({k: res[k] for k in
          ("variant", "S", "bits", "trainable_params", "best_val_acc", "best_step",
           "acc", "causal_max_abs", "future_sensitivity", "runtime_s")}, ensure_ascii=False))


def summarize(d: Path):
    import glob
    rows = []
    for f in glob.glob(str(d / "e7*.json")):
        x = json.load(open(f))
        rows.append(x)
    rows.sort(key=lambda r: (r["variant"], r.get("S") or 0, r["bits"]))
    lags = sorted({int(k) for r in rows for k in r["acc"]})
    print("| arm | params | " + " | ".join(f"l={L}" for L in lags) +
          " | 鎺у埗 rand/shuf/nocue (l=max) |")
    print("|---|---|" + "---|" * len(lags) + "---|")
    for r in rows:
        arm = r["variant"] + (f"-S{r['S']}" if r.get("S") else "") + f"-B{r['bits']}"
        cells = []
        for L in lags:
            v = r["acc"].get(str(L))
            cells.append("-" if v is None else f"{v:.3f}")
        mx = str(max(lags))
        c = r["controls"]
        ctl = "/".join(f"{c[k].get(mx, float('nan')):.3f}" for k in ("randcue", "shuffle", "nocue"))
        print(f"| {arm} | {r['trainable_params']/1e3:.0f}k | " + " | ".join(cells) + f" | {ctl} |")


if __name__ == "__main__":
    main()
