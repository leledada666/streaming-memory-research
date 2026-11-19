# -*- coding: utf-8 -*-
"""Content-addressed vs position-addressed recall on frozen Kronos-small.

Stream: m_t = [onehot(k_t) (K=64), c_t (+-1), z_t ~ N(0,1)].
Task pos:     y_t = c_{t-L}; query = learned fixed token (task marker), read at t.
Task content: q_t = onehot(k_s), s ~ U[0,t-1]; y_t = c_s; must key-match, no PE needed.
Arms: bypass (gate*MLP(m_t) into every layer input, query appended to m_t at t),
      crossattn (Q=Wq(q_t)+PE?, KV=Wk([k_s,c_s])+PE? for s<=t, strict causal),
      gru (GRU over [m_t,q_t], head reads state).
Head: MLP([h_norm*8, adapter]) -> 1 logit.
Read-only on E:\\rwkv7; all outputs under the temp dir.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(r"E:\rwkv7")
OUTDIR = Path(r"C:\Users\lele\AppData\Local\Temp\opencode\ca_vs_pa")
OUTDIR.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "binance" / "klines_1m" / "bars_1m_of_3m.csv"
TOK = ROOT / "runs" / "e6_long_2026-10-20" / "e5_tokens.npz"
sys.path.insert(0, str(ROOT / "kronos_src"))
from model import Kronos  # noqa: E402

K = 64
ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, default=32)
ap.add_argument("--B", type=int, default=4)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--eval_every", type=int, default=50)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--seeds", default="0")
ap.add_argument("--only", default="", help="comma list of run names (default: smoke set)")
ap.add_argument("--valB", type=int, default=8)
ap.add_argument("--nval", type=int, default=4)
ap.add_argument("--out", default="")
a = ap.parse_args()
assert a.device == "cpu" or torch.cuda.is_available(), "cuda requested but unavailable"
dev = torch.device(a.device)
torch.set_num_threads(8 if a.device == "cpu" else 1)
SEEDS = [int(x) for x in a.seeds.split(",") if x.strip() != ""]

# ---------------- data ----------------
toks = np.load(TOK)
s1_all = torch.tensor(toks["S1"])
s2_all = torch.tensor(toks["S2"])
df = pd.read_csv(DATA, usecols=["ts"])
N = len(df)
assert len(s1_all) == N, (len(s1_all), N)
ts = pd.to_datetime(df["ts"], unit="s", utc=True)
STAMP = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1).astype(np.float32)
st_all = torch.tensor(STAMP)
i1, i2 = int(N * 0.70), int(N * 0.85)
S1, S2, ST = s1_all.to(dev), s2_all.to(dev), st_all.to(dev)

KRONOS = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
for p in KRONOS.parameters():
    p.requires_grad = False
H = KRONOS.d_model
NL = len(KRONOS.transformer)
LAYERS = list(KRONOS.transformer)
assert all(not p.requires_grad for p in KRONOS.parameters())


def onehot(k: np.ndarray) -> np.ndarray:
    o = np.zeros(k.shape + (K,), np.float32)
    np.put_along_axis(o, k[..., None], 1.0, -1)
    return o


def make_batch(g: np.random.Generator, lo: int, hi: int, T: int, B: int, task: str, L: int) -> dict:
    starts = g.integers(lo, hi - T, B)
    idx = torch.as_tensor(starts[:, None] + np.arange(T)[None, :], device=dev)
    c = (g.integers(0, 2, (B, T)) * 2 - 1).astype(np.float32)
    z = g.standard_normal((B, T)).astype(np.float32)
    k = g.integers(0, K, (B, T))
    m = np.concatenate([onehot(k), c[..., None], z[..., None]], -1).astype(np.float32)
    q = np.zeros((B, T, K), np.float32)
    y = np.zeros((B, T), np.float32)
    valid = np.zeros((B, T), bool)
    lag = np.full((B, T), -1, np.int32)
    bg = np.arange(B)[:, None]
    if task == "pos":
        if L == 0:
            valid[:] = True
            y[:] = (c > 0)
            lag[:] = 0
        else:
            valid[:, L:] = True
            y[:, L:] = (c[:, : T - L] > 0)
            lag[:, L:] = L
    else:  # content: s strictly < t
        ss = np.stack([g.integers(0, t, B) for t in range(1, T)], 1)
        q[:, 1:, :] = onehot(k[bg, ss])
        y[:, 1:] = (c[bg, ss] > 0)
        lag[:, 1:] = np.arange(1, T)[None, :] - ss
        valid[:, 1:] = True
    out = dict(s1=S1[idx], s2=S2[idx], st=ST[idx],
               m=torch.tensor(m, device=dev), q=torch.tensor(q, device=dev),
               y=torch.tensor(y, device=dev), valid=torch.tensor(valid, device=dev),
               lag=torch.tensor(lag, device=dev), k=torch.tensor(k, device=dev),
               c=torch.tensor(c, device=dev))
    return out


# ---------------- arms ----------------
class Net(nn.Module):
    def __init__(self, arm: str, task: str, pe: bool = False):
        super().__init__()
        self.arm, self.task, self.use_pe = arm, task, pe
        self.qpos = nn.Parameter(torch.zeros(K)) if task == "pos" else None
        self.adim = H
        if arm == "bypass":
            self.enc = nn.ModuleList([
                nn.Sequential(nn.Linear(K + 66, 64), nn.GELU(), nn.Linear(64, H)) for _ in LAYERS])
            self.gate = nn.Parameter(torch.full((len(LAYERS),), -2.0))
            self.enc_head = nn.Sequential(nn.Linear(K + 66, H), nn.GELU(), nn.Linear(H, H))
        elif arm == "crossattn":
            self.nh = 4
            self.hd = H // self.nh
            self.qp = nn.Linear(K, H)
            self.kp = nn.Linear(K + 1, H)
            self.vp = nn.Linear(K + 1, H)
            self.op = nn.Linear(H, H)
            self.pe = nn.Parameter(torch.zeros(a.T, H)) if pe else None
        elif arm == "gru":
            self.gru = nn.GRU(K + 66, 128, batch_first=True)
            self.adim = 128
        self.head = nn.Sequential(nn.Linear(H + self.adim, 128), nn.GELU(), nn.Linear(128, 1))
        self._causal_call = 0

    def q_seq(self, q: torch.Tensor) -> torch.Tensor:
        if self.task == "content":
            return q
        return self.qpos.view(1, 1, K).expand(q.shape[0], q.shape[1], K)

    def attn_adapter(self, kv1: torch.Tensor, qs: torch.Tensor) -> torch.Tensor:
        B, T, _ = qs.shape
        Q = self.qp(qs)
        Kk = self.kp(kv1)
        V = self.vp(kv1)
        if self.pe is not None:
            Q = Q + self.pe[:T]
            Kk = Kk + self.pe[:T]
        Q = Q.view(B, T, self.nh, self.hd).transpose(1, 2)
        Kk = Kk.view(B, T, self.nh, self.hd).transpose(1, 2)
        V = V.view(B, T, self.nh, self.hd).transpose(1, 2)
        sc = (Q @ Kk.transpose(-2, -1)) / (self.hd ** 0.5)
        ii = torch.arange(T, device=qs.device)
        allow = (ii[None, :] <= ii[:, None]) if self.task == "pos" else (ii[None, :] < ii[:, None])
        if not allow[0, 0]:
            allow = allow.clone()
            allow[0, 0] = True  # t=0 has no past key; keep self so softmax is defined (position 0 is never a target)
        mask = torch.zeros(T, T, dtype=torch.bool, device=qs.device)
        mask[~allow] = True
        self.last_mask = allow
        sc = sc.masked_fill(mask, float("-inf"))
        ad = (sc.softmax(-1) @ V).transpose(1, 2).reshape(B, T, H)
        return self.op(ad)

    def logits(self, b: dict, m_override: torch.Tensor | None = None) -> torch.Tensor:
        m = b["m"] if m_override is None else m_override
        x = KRONOS.embedding([b["s1"], b["s2"]]) + KRONOS.time_emb(b["st"])
        qs = self.q_seq(b["q"])
        maug = torch.cat([m, qs], -1)
        adapter = None
        if self.arm == "bypass":
            for i, layer in enumerate(LAYERS):
                u = self.enc[i](maug)
                u = u / (u.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)
                x = x + torch.sigmoid(self.gate[i]) * u
                x = layer(x)
            h = KRONOS.norm(x)
            adapter = self.enc_head(maug)  # E6-style direct m->head path (parity with other arms)
        else:
            for layer in LAYERS:
                x = layer(x)
            h = KRONOS.norm(x)
            if self.arm == "crossattn":
                kv1 = torch.cat([onehot_t(b["k"]), b["c"].unsqueeze(-1)], -1)
                adapter = self.attn_adapter(kv1, qs)
            else:
                adapter = self.gru(maug)[0]
        hn = h / (h.norm(dim=-1, keepdim=True) + 1e-6) * 8
        return self.head(torch.cat([hn, adapter], -1)).squeeze(-1)


def onehot_t(k: torch.Tensor) -> torch.Tensor:
    return F.one_hot(k.long(), K).float()


def masked_acc(lg: torch.Tensor, b: dict) -> tuple[float, int]:
    v = b["valid"]
    cor = ((lg > 0).float() == b["y"])[v].sum().item()
    return cor / max(1, int(v.sum())), int(v.sum())


def masked_loss(lg, b):
    l = F.binary_cross_entropy_with_logits(lg, b["y"], reduction="none")
    v = b["valid"]
    return (l * v).sum() / v.sum().clamp(min=1)


# ---------------- causality / leakage audits ----------------
def audit_causality(net: Net, task: str, L: int, T: int) -> dict:
    g = np.random.default_rng(12345)
    b = make_batch(g, 0, i1, T, 4, task, L)
    t0 = T // 2
    with torch.no_grad():
        lg = net.logits(b)
        b2 = dict(b)
        m2 = b["m"].clone()
        m2[:, t0 + 1:, :] = torch.randn_like(m2[:, t0 + 1:, :])
        lg2 = net.logits(b2, m_override=m2)
    d = (lg - lg2)[:, : t0 + 1].abs().max().item()
    return {"t0": t0, "max_abs_logit_diff_past": d}


def audit_leak(T: int, L: int) -> dict:
    """Does q_t=onehot(k_s) carry any information about c_s beyond the key itself?
    One sample per (sequence, matched position s); split BY SEQUENCE so repeated
    queries of the same key cannot be memorised across the split."""
    g = np.random.default_rng(7)
    b = make_batch(g, 0, i1, T, 64, "content", L)
    k = b["k"].cpu().numpy()[:, : T - 1]
    c = b["c"].cpu().numpy()[:, : T - 1]
    nb = k.shape[0]
    half = nb // 2
    Xtr = torch.tensor(onehot(k[:half]).reshape(-1, K))
    Ytr = torch.tensor((c[:half].reshape(-1) > 0).astype(np.float32))
    Xte = torch.tensor(onehot(k[half:]).reshape(-1, K))
    Yte = torch.tensor((c[half:].reshape(-1) > 0).astype(np.float32))
    w = torch.zeros(K)
    bias = torch.zeros(())
    for _ in range(300):
        p = torch.sigmoid(Xtr @ w + bias)
        err = p - Ytr
        w -= 0.5 * (Xtr.t() @ err / len(Ytr))
        bias -= 0.5 * err.mean()
    acc = (((Xte @ w + bias) > 0).float() == Yte).float().mean().item()
    bq = b["q"].reshape(-1, K)
    bv = b["valid"].reshape(-1)
    qq = bq[bv]
    return {"q_dim": K, "onehot_sum_min": float(qq.sum(1).min()), "onehot_sum_max": float(qq.sum(1).max()),
            "onehot_binary": bool(((qq == 0) | (qq == 1)).all()),
            "probe_acc_heldout_q_to_c": round(acc, 3), "probe_chance": 0.5,
            "n_probe_train": int(len(Ytr)), "n_probe_test": int(len(Yte)),
            "split": "by_sequence"}


def shortcut_baseline(task: str, L: int) -> dict:
    """Linear readout of the prefix mean of c (+t): the 'mean-pool' shortcut ceiling.
    Any arm that pools the prefix reads the target element with weight 1/(t+1), so this
    baseline bounds how much accuracy is obtainable without true addressing."""
    g = np.random.default_rng(555)

    def feat(b):
        c = b["c"].cpu().numpy()
        cs = np.cumsum(c, 1) / np.arange(1, c.shape[1] + 1)[None, :]
        tt = np.broadcast_to(np.arange(c.shape[1]) / float(c.shape[1]), c.shape)
        X = np.stack([cs, tt], -1)
        v = b["valid"].cpu().numpy()
        return X[v], b["y"].cpu().numpy()[v]

    def gather(lo, hi, nb):
        Xs, Ys = [], []
        for _ in range(nb):
            X, Y = feat(make_batch(g, lo, hi, a.T, 8, task, L))
            Xs.append(X)
            Ys.append(Y)
        return torch.tensor(np.concatenate(Xs)).float(), torch.tensor(np.concatenate(Ys)).float()

    Xtr, Ytr = gather(0, i1, 40)
    Xva, Yva = gather(i1, i2, 8)
    w = torch.zeros(Xtr.shape[1])
    b0 = torch.zeros(())
    for _ in range(400):
        p = torch.sigmoid(Xtr @ w + b0)
        err = p - Ytr
        w -= 0.5 * (Xtr.t() @ err / len(Ytr))
        b0 -= 0.5 * err.mean()
    acc = (((Xva @ w + b0) > 0).float() == Yva).float().mean().item()
    return {"val_acc_prefix_mean": round(acc, 4), "n_val": int(len(Yva))}


def lag_hist(T: int, L: int, nb: int = 50) -> dict:
    g = np.random.default_rng(3)
    b = make_batch(g, i1, i2, T, nb, "content", L)
    lg = b["lag"][b["valid"]].cpu().numpy()
    vals = np.bincount(lg, minlength=T)
    return {"min": int(lg.min()), "max": int(lg.max()), "mean": round(float(lg.mean()), 2),
            "frac_lag1": round(float((lg == 1).mean()), 3),
            "frac_lag_lt8": round(float((lg < 8).mean()), 3),
            "hist": {int(i): int(v) for i, v in enumerate(vals) if v}}


# ---------------- run ----------------
def run(arm: str, task: str, pe: bool, L: int, seed: int) -> dict:
    tag = name_of((arm, task, pe, L))
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    net = Net(arm, task, pe).to(dev)
    params = [p for p in net.parameters() if p.requires_grad]
    ntr = sum(p.numel() for p in params)
    opt = torch.optim.Adam(params, lr=a.lr)
    vg = np.random.default_rng(10_000 + seed)
    VAL = [make_batch(vg, i1, i2, a.T, a.valB, task, L) for _ in range(a.nval)]
    tg = np.random.default_rng(20_000 + seed)
    TEST = [make_batch(tg, i2, N, a.T, a.valB, task, L) for _ in range(a.nval)]
    audit = audit_causality(net, task, L, a.T)
    vis = None
    if arm == "crossattn":
        with torch.no_grad():
            _ = net.logits(VAL[0])
        allow = net.last_mask
        t0 = min(a.T - 1, 8)
        vis = {"query_pos": t0, "visible_keys": [int(i) for i in torch.nonzero(allow[t0], as_tuple=False).flatten().tolist()]}
    print(f"[{tag} s{seed}] params {ntr} causal {json.dumps(audit)} vis {json.dumps(vis)}", flush=True)
    curve, best, best_step, best_state = [], -1.0, 0, None
    t0 = time.time()
    for it in range(1, a.steps + 1):
        net.train()
        b = make_batch(g, 0, i1, a.T, a.B, task, L)
        lg = net.logits(b)
        loss = masked_loss(lg, b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % a.eval_every == 0 or it == a.steps:
            net.eval()
            cor = n = 0
            with torch.no_grad():
                for vb in VAL:
                    c_, n_ = masked_acc(net.logits(vb), vb)
                    cor += c_ * n_
                    n += n_
            acc = cor / n
            gates = [round(float(torch.sigmoid(g_).item()), 3) for g_ in net.gate] if arm == "bypass" else None
            curve.append({"step": it, "val_acc": round(acc, 4), "loss": round(float(loss.item()), 4), "gates": gates})
            if acc > best:
                best, best_step = acc, it
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            print(f"[{tag} s{seed}] step {it:4d} loss {loss.item():.4f} val {acc:.3f} gates {gates}", flush=True)
    net.load_state_dict(best_state)
    net.eval()
    if arm == "crossattn":
        # read-out audit: does the model read c from KV at all?
        tb = TEST[0]
        with torch.no_grad():
            lg_ok = net.logits(tb)
            fake = dict(tb)
            fake["c"] = (torch.randint(0, 2, tb["c"].shape, device=dev).float() * 2 - 1)
            lg_bad = net.logits(fake)
        v = tb["valid"]
        audit["acc_orig"] = round(((lg_ok > 0).float() == tb["y"])[v].float().mean().item(), 3)
        audit["acc_kv_c_shuffled"] = round(((lg_bad > 0).float() == tb["y"])[v].float().mean().item(), 3)
    cor = n = 0
    with torch.no_grad():
        for tb in TEST:
            c_, n_ = masked_acc(net.logits(tb), tb)
            cor += c_ * n_
            n += n_
    test_acc = cor / n
    res = {"run": tag, "arm": arm, "task": task, "pe": pe if arm == "crossattn" else None, "lag": L, "seed": seed,
           "T": a.T, "B": a.B, "steps": a.steps, "trainable_params": int(ntr),
           "best_val_acc": round(best, 4), "best_step": best_step, "test_acc": round(test_acc, 4),
           "causal_check": audit, "attn_visibility": vis, "curve": curve,
           "runtime_s": round(time.time() - t0, 1)}
    print(f"RESULT {json.dumps({k: res[k] for k in ['run','seed','trainable_params','best_val_acc','best_step','test_acc','runtime_s']})}", flush=True)
    return res


SMOKE = [("bypass", "pos", False, 5), ("bypass", "pos", False, 0), ("bypass", "content", False, 0),
         ("crossattn", "pos", False, 5), ("crossattn", "pos", True, 5), ("crossattn", "pos", False, 0),
         ("crossattn", "content", False, 0), ("crossattn", "content", True, 0),
         ("gru", "content", False, 0)]
FULL = [("bypass", "pos", False, 5), ("bypass", "content", False, 0),
        ("crossattn", "pos", False, 5), ("crossattn", "pos", True, 5),
        ("crossattn", "content", False, 0), ("crossattn", "content", True, 0),
        ("gru", "content", False, 0)]


def name_of(c):
    arm, task, pe, L = c
    if arm == "crossattn":
        return f"crossattn_{task}" + (f"_L{L}" if task == "pos" else "") + ("_pe" if pe else "_nope")
    return f"{arm}_{task}" + (f"_L{L}" if task == "pos" else "")


def main():
    cfg = FULL if a.steps > 1000 else SMOKE
    wanted = {x.strip() for x in a.only.split(",") if x.strip()}
    results, lags = [], {}
    for c in cfg:
        if wanted and name_of(c) not in wanted:
            continue
        for sd in SEEDS:
            results.append(run(*c, sd))
    if not wanted or "bypass_content" in wanted:
        lags["content"] = lag_hist(a.T, 0)
        lags["leak"] = audit_leak(a.T, 0)
        print("LAG", json.dumps(lags["content"]), flush=True)
        print("LEAK", json.dumps(lags["leak"]), flush=True)
    out = {"config": vars(a), "T": a.T, "B": a.B, "steps": a.steps, "results": results, "lags": lags}
    if not wanted:
        bl = {"pos_L5": shortcut_baseline("pos", 5), "pos_L0": shortcut_baseline("pos", 0),
              "content": shortcut_baseline("content", 0)}
        out["baselines"] = bl
        print("BASELINE", json.dumps(bl), flush=True)
    p = OUTDIR / (a.out or f"smoke_ca_pa_T{a.T}_B{a.B}_{a.steps}.json")
    json.dump(out, open(p, "w"), indent=1)
    print("WROTE", p, flush=True)


if __name__ == "__main__":
    main()