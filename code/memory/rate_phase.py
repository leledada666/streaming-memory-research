# -*- coding: utf-8 -*-
"""Rate phase diagram (E7): r observations per bar on real BTC 1m tokens.

Per bar t we draw r sub-observations m_{t,j} = [c_{t,j} (iid +-1), z_{t,j} ~ N(0,1)].
Task A (aggregate):  y_t = 1[ sum_j c_{t-L,j} > 0 ]   (zero-sum bars dropped; chance 0.5)
Task B (sub-position): y_t = 1[ c_{t-L, r-1} > 0 ]
Frozen Kronos-small carrier (S1,S2 from E6 tokens + STAMP); only the read/inject
modules and the 2H->128->1 head are trained.

Arms
  bypass_mean : bar-pool mean -> 2 dims -> gated inject at all 8 layer inputs + head
  bypass_last : bar-pool last -> 2 dims -> same write path
  crossattn   : Q from h_t, K/V from the raw rL window (relative slot table), 2 blocks
  gru         : GRU over the raw r*T stream, hidden at each bar -> inject at all sites

Controls: L=0 (positive, window = current bar), shuffle-time tokens, random query (crossattn).

Smoke (CPU): 200 steps, T=16 B=4 / T=32 B=2, seed 0.  Full (GPU): T=96 B=32 cap=15000.
"""
from __future__ import annotations
import argparse, json, math, sys, time, traceback
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
TOK = ROOT / "runs" / "e6_long_2026-10-20" / "e5_tokens.npz"
OUT = Path(r"C:\Users\lele\AppData\Local\Temp\opencode\rate_phase_out")

ARMS = ["bypass_mean", "bypass_last", "crossattn", "gru"]
H = 512
NL = 8

# ---------------------------------------------------------------- worker state
M = None
S1T = S2T = STT = None
N = I1 = I2 = 0
DEV = "cpu"
THREADS = 6


def init_worker(threads=6, dev="cpu"):
    global M, S1T, S2T, STT, N, I1, I2, DEV, THREADS
    THREADS, DEV = threads, dev
    torch.set_num_threads(threads)
    m = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    df = pd.read_csv(DATA, usecols=["ts"])
    ts = pd.to_datetime(df["ts"], unit="s", utc=True)
    stamp = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1).astype(np.float32)
    z = np.load(TOK)
    M = m
    S1T = torch.tensor(z["S1"], dtype=torch.long, device=dev)
    S2T = torch.tensor(z["S2"], dtype=torch.long, device=dev)
    STT = torch.tensor(stamp, device=dev)
    N = len(df)
    I1, I2 = int(N * 0.70), int(N * 0.85)
    return m


# ---------------------------------------------------------------- modules
def mlp(i, w, o):
    return nn.Sequential(nn.Linear(i, w), nn.GELU(), nn.Linear(w, o))


class XBlock(nn.Module):
    """Single-head cross-attention over a fixed relative-slot window."""

    def __init__(s, h, d, nslots):
        super().__init__()
        s.ln = nn.LayerNorm(h)
        s.q = nn.Linear(h, d)
        s.k = nn.Linear(2, d)
        s.v = nn.Linear(2, d)
        s.pos = nn.Parameter(torch.zeros(nslots, d))
        s.out = nn.Linear(d, h)
        s.d = d

    def forward(s, h, keys, randq=False):
        q = s.q(torch.randn_like(h) if randq else s.ln(h))
        k = s.k(keys) + s.pos[: keys.shape[-2]]
        v = s.v(keys)
        a = (q.unsqueeze(2) * k).sum(-1) / math.sqrt(s.d)
        a = a.softmax(-1)
        u = (a.unsqueeze(-1) * v).sum(2)
        return h + s.out(u), u


class Net(nn.Module):
    def __init__(s, arm, r, lag, T):
        super().__init__()
        s.arm, s.r, s.lag, s.T = arm, r, lag, T
        s.nslots = r * max(lag, 1)
        s.g = nn.Parameter(torch.full((NL,), -2.0))
        if arm in ("bypass_mean", "bypass_last"):
            s.enc = nn.ModuleList([mlp(2, 64, H) for _ in range(NL)])
            s.head_enc = mlp(2, H, H)
        elif arm == "gru":
            s.gru = nn.GRU(2, 64, 1, batch_first=True)
            s.enc = nn.ModuleList([mlp(64, 64, H) for _ in range(NL)])
            s.head_enc = mlp(64, 256, H)
        elif arm == "crossattn":
            s.blocks = nn.ModuleList([XBlock(H, 192, s.nslots) for _ in range(2)])
            s.head_proj = nn.Linear(192, H)
        else:
            raise ValueError(arm)
        s.pi = nn.Sequential(nn.Linear(2 * H, 128), nn.GELU(), nn.Linear(128, 1))

    # window over the raw observations: L>0 -> bars [t-L, t-1]; L==0 -> bar t
    def windows(s, obs):
        B, T, r, _ = obs.shape
        L = s.lag
        if L == 0:
            return obs
        flat = obs.reshape(B, T * r, 2)
        w = flat.unfold(1, L * r, r).permute(0, 1, 3, 2)  # start bar k -> bars k..k+L-1
        keys = torch.zeros(B, T, L * r, 2, device=obs.device, dtype=obs.dtype)
        keys[:, L:] = w[:, : T - L]
        return keys

    def summarise(s, obs):
        B, T, r, _ = obs.shape
        if s.arm == "bypass_mean":
            return obs.mean(2), None
        if s.arm == "bypass_last":
            return obs[:, :, r - 1], None
        if s.arm == "gru":
            o, _ = s.gru(obs.reshape(B, T * r, 2))
            return o[:, r - 1 :: r], None  # hidden after last sub-obs of each bar
        return None, None

    def forward(s, s1, s2, st, obs, randq=False):
        B, T, r, _ = obs.shape
        pool, _ = s.summarise(obs)
        x = M.embedding([s1, s2]) + M.time_emb(st)
        if s.arm in ("bypass_mean", "bypass_last", "gru"):
            for i, layer in enumerate(M.transformer):
                u = s.enc[i](pool)
                u = u / (u.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)
                x = x + torch.sigmoid(s.g[i]) * u
                x = layer(x)
            h = M.norm(x)
            hm = s.head_enc(pool)
        else:
            for layer in M.transformer:
                x = layer(x)
            h = M.norm(x)
            keys = s.windows(obs)
            u = None
            for blk in s.blocks:
                h, u = blk(h, keys, randq=randq)
            hm = s.head_proj(u)
        hn = h / (h.norm(dim=-1, keepdim=True) + 1e-6) * 8
        return s.pi(torch.cat([hn, hm], -1)).squeeze(-1)


# ---------------------------------------------------------------- data
def make_batch(lo, hi, B, g, T, r, L, task, shuffle=False):
    st = g.integers(lo, hi - T, B)
    idx = torch.tensor(st[:, None] + np.arange(T), device=DEV)
    if shuffle:  # scramble the carrier's time order, keep m/y aligned
        perm = torch.tensor(np.stack([g.permutation(T) for _ in range(B)]), device=DEV)
        idx = idx.gather(1, perm)
    c = torch.tensor(g.choice([-1.0, 1.0], size=(B, T, r)).astype(np.float32), device=DEV)
    z = torch.tensor(g.standard_normal((B, T, r)).astype(np.float32), device=DEV)
    obs = torch.stack([c, z], -1)
    y = torch.zeros(B, T, device=DEV)
    mask = torch.zeros(B, T, device=DEV)
    if L > 0:
        src = c[:, : T - L]
        if task == "A":
            ssum = src.sum(-1)
            m = ssum != 0
            y[:, L:] = (ssum > 0).float()
            mask[:, L:] = m.float()
        else:
            y[:, L:] = (src[..., r - 1] > 0).float()
            mask[:, L:] = 1.0
    else:
        if task == "A":
            ssum = c.sum(-1)
            m = ssum != 0
            y = (ssum > 0).float()
            mask = m.float()
        else:
            y = (c[..., r - 1] > 0).float()
            mask = torch.ones_like(y)
    return S1T[idx], S2T[idx], STT[idx], obs, y, mask


def eval_sets(T, r, L, task):
    vg = np.random.default_rng(999)
    tg = np.random.default_rng(777)
    VAL = [make_batch(I1, I2, 16, vg, T, r, L, task) for _ in range(4)]
    TEST = [make_batch(I2, N, 16, tg, T, r, L, task) for _ in range(8)]
    return VAL, TEST


def evaluate(net, SET, L, randq=False):
    net.eval()
    cor = tot = 0
    with torch.no_grad():
        for s1, s2, st, obs, y, m in SET:
            lg = net(s1, s2, st, obs, randq=randq)
            sl = slice(L, None) if L > 0 else slice(None)
            pred = (lg[:, sl] > 0).float()
            mm = m[:, sl]
            cor += ((pred == y[:, sl]).float() * mm).sum().item()
            tot += mm.sum().item()
    net.train()
    return cor / max(tot, 1.0), int(tot)


def base_rate(SET, L):
    ones = tot = 0
    for *_, y, m in SET:
        sl = slice(L, None) if L > 0 else slice(None)
        ones += (y[:, sl] * m[:, sl]).sum().item()
        tot += m[:, sl].sum().item()
    p = ones / max(tot, 1)
    return max(p, 1 - p)


# ---------------------------------------------------------------- causes check
def causality_check(net, T, r, L, task):
    Tck = max(T, L + 16)
    g = np.random.default_rng(4242)
    s1, s2, st, obs, y, m = make_batch(0, min(I1, 5000), 2, g, Tck, r, L, task)
    t0 = Tck // 2

    def lg(o):
        with torch.no_grad():
            return net(s1, s2, st, o)

    base = lg(obs)
    g2 = np.random.default_rng(5151)
    o2 = obs.clone()
    o2[:, t0 + 1:] = torch.stack([
        torch.tensor(g2.choice([-1.0, 1.0], size=(2, Tck - t0 - 1, r)).astype(np.float32), device=DEV),
        torch.tensor(g2.standard_normal((2, Tck - t0 - 1, r)).astype(np.float32), device=DEV)], -1)
    past = (base[:, : t0 + 1] - lg(o2)[:, : t0 + 1]).abs().max().item()

    # current bar must not enter a L>0 cross-attn window
    o3 = obs.clone()
    o3[:, t0] = torch.stack([
        torch.tensor(g2.choice([-1.0, 1.0], size=(2, r)).astype(np.float32), device=DEV),
        torch.tensor(g2.standard_normal((2, r)).astype(np.float32), device=DEV)], -1)
    cur = (base[:, t0] - lg(o3)[:, t0]).abs().max().item()
    fut = 0.0
    if L > 0 and t0 + L < Tck:
        fut = (base[:, t0 + L] - lg(o3)[:, t0 + L]).abs().max().item()
    return {"past_future_perturb_maxdiff": past, "current_bar_self_maxdiff": cur,
            "bar_t0_effect_at_t0_plus_L": fut}


# ---------------------------------------------------------------- one run
def run_one(cfg):
    t_start = time.time()
    init_worker(cfg.get("threads", THREADS), cfg.get("device", "cpu"))
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(1000 + cfg["seed"])
    T, r, L, task, arm = cfg["T"], cfg["r"], cfg["lag"], cfg["task"], cfg["arm"]
    shuffle, randq = cfg.get("control") == "shuffle", cfg.get("control") == "randq"
    net = Net(arm, r, L, T).to(DEV)
    nparam = sum(p.numel() for p in net.parameters())
    data_ok = True
    gg = np.random.default_rng(31)
    s1, s2, st, obs, y, mask = make_batch(0, 2000, 2, gg, T, r, L, task)
    if L > 0:
        c = obs[..., 0]
        ref = (c[:, : T - L].sum(-1) > 0).float() if task == "A" else (c[:, : T - L, r - 1] > 0).float()
        data_ok = bool(torch.equal(ref, y[:, L:]))
    else:
        c = obs[..., 0]
        ref = (c.sum(-1) > 0).float() if task == "A" else (c[..., r - 1] > 0).float()
        data_ok = bool(torch.equal(ref, y))
    ck = causality_check(net, T, r, L, task)

    VAL, TEST = eval_sets(T, r, L, task)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.get("lr", 1e-3))
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    curve, best, best_step, best_state, stop = [], -1.0, 0, None, "cap"
    ms_acc, n_ms = 0.0, 0
    for it in range(1, cfg["steps"] + 1):
        s1, s2, st, obs, y, mask = make_batch(0, I1, cfg["B"], rng, T, r, L, task, shuffle)
        t0 = time.perf_counter()
        lg = net(s1, s2, st, obs, randq=randq)
        sl = slice(L, None) if L > 0 else slice(None)
        l = lossf(lg[:, sl], y[:, sl])
        mm = mask[:, sl]
        loss = (l * mm).sum() / mm.sum().clamp_min(1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ms_acc += (time.perf_counter() - t0) * 1000
        n_ms += 1
        if it % cfg.get("eval_every", 50) == 0:
            acc, ntg = evaluate(net, VAL, L, randq)
            curve.append({"step": it, "val_acc": acc, "loss": float(loss)})
            if acc > best + 1e-3:
                best, best_step, best_state = acc, it, {k: v.detach().clone() for k, v in net.state_dict().items()}
            if it >= cfg.get("min_steps", 10 ** 9) and it - best_step >= cfg.get("patience", 10 ** 9):
                stop = "early_stop"
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    test_acc, ntest = evaluate(net, TEST, L, randq)
    val_acc = best if best >= 0 else evaluate(net, VAL, L, randq)[0]
    ms_step = ms_acc / max(n_ms, 1)
    net.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(3):
            net(s1, s2, st, obs, randq=randq)
        ms_fwd = (time.perf_counter() - t0) * 1000 / 3
    res = {"arm": arm, "task": task, "r": r, "lag": L, "T": T, "B": cfg["B"],
           "control": cfg.get("control", "none"), "seed": cfg["seed"],
           "read_len_obs": (L * r if L > 0 else r), "seq_bars": T,
           "trainable_params": int(nparam),
           "val_acc": round(float(val_acc), 4), "test_acc": round(float(test_acc), 4),
           "base_majority": round(float(base_rate(TEST, L)), 4), "n_test_targets": ntest,
           "best_step": best_step, "steps_run": it, "stop_reason": stop,
           "ms_step_train": round(ms_step, 1), "ms_step_fwd": round(ms_fwd, 1),
           "causal": ck, "data_label_ok": data_ok, "curve": curve,
           "runtime_s": round(time.time() - t_start, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / ("rate_%s_%s_r%d_L%d_T%d_%s_s%d.json" % (arm, task, r, L, T, res["control"], cfg["seed"]))
    json.dump(res, open(fn, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ["arm", "task", "r", "lag", "T", "control", "test_acc",
                                          "base_majority", "ms_step_train", "runtime_s"]}), flush=True)
    return res


# ---------------------------------------------------------------- grids
def grid_smoke():
    runs = []
    for T, B in ((16, 4),):
        for L in (5, 10):
            for r in (1, 4):
                for task in ("A", "B"):
                    for arm in ARMS:
                        runs.append(dict(arm=arm, task=task, r=r, lag=L, T=T, B=B, steps=200,
                                         seed=0, control="none", eval_every=50))
    for L in (10,):
        for task in ("A", "B"):
            for arm in ARMS:
                runs.append(dict(arm=arm, task=task, r=4, lag=L, T=32, B=2, steps=200,
                                 seed=0, control="none", eval_every=50))
    for task in ("A", "B"):
        for arm in ARMS:
            runs.append(dict(arm=arm, task=task, r=4, lag=0, T=16, B=4, steps=200,
                             seed=0, control="none", eval_every=50))
    for arm, task in (("bypass_mean", "A"), ("bypass_last", "B"), ("crossattn", "A"), ("gru", "A")):
        runs.append(dict(arm=arm, task=task, r=4, lag=10, T=16, B=4, steps=200,
                         seed=0, control="shuffle", eval_every=50))
    runs.append(dict(arm="crossattn", task="A", r=4, lag=10, T=16, B=4, steps=200,
                     seed=0, control="randq", eval_every=50))
    return runs


def grid_full():
    runs = []
    for L in (5, 10):
        for r in (1, 4):
            for task in ("A", "B"):
                for arm in ARMS:
                    for seed in (0, 1, 2):
                        runs.append(dict(arm=arm, task=task, r=r, lag=L, T=96, B=32, steps=15000,
                                         seed=seed, control="none", eval_every=100,
                                         min_steps=2000, patience=800, lr=1e-3))
    for task in ("A", "B"):
        for arm in ARMS:
            runs.append(dict(arm=arm, task=task, r=4, lag=0, T=96, B=32, steps=15000,
                             seed=0, control="none", eval_every=100,
                             min_steps=2000, patience=800, lr=1e-3))
    for arm, task in (("bypass_mean", "A"), ("bypass_last", "B"), ("crossattn", "A"), ("gru", "A")):
        runs.append(dict(arm=arm, task=task, r=4, lag=10, T=96, B=32, steps=15000,
                         seed=0, control="shuffle", eval_every=100,
                         min_steps=2000, patience=800, lr=1e-3))
    runs.append(dict(arm="crossattn", task="A", r=4, lag=10, T=96, B=32, steps=15000,
                     seed=0, control="randq", eval_every=100,
                     min_steps=2000, patience=800, lr=1e-3))
    return runs


def fmt_table(rows):
    head = "| arm | task | r | L | T | ctrl | acc | base | read | params | ms/step |"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for x in sorted(rows, key=lambda z: (z["T"], z["lag"], z["r"], z["task"], z["control"], z["arm"])):
        lines.append("| %s | %s | %d | %d | %d | %s | %.3f | %.3f | %d | %.0fk | %.0f |" % (
            x["arm"], x["task"], x["r"], x["lag"], x["T"], x["control"], x["test_acc"],
            x["base_majority"], x["read_len_obs"], x["trainable_params"] / 1000, x["ms_step_train"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="", choices=["", "smoke", "full"])
    ap.add_argument("--arm", default="bypass_mean", choices=ARMS)
    ap.add_argument("--task", default="A", choices=["A", "B"])
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--lag", type=int, default=10)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control", default="none", choices=["none", "shuffle", "randq"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--min_steps", type=int, default=10 ** 9)
    ap.add_argument("--patience", type=int, default=10 ** 9)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    cfg_single = dict(arm=a.arm, task=a.task, r=a.r, lag=a.lag, T=a.T, B=a.B, steps=a.steps,
                      seed=a.seed, control=a.control, eval_every=a.eval_every,
                      min_steps=a.min_steps, patience=a.patience, lr=a.lr,
                      device=a.device, threads=a.threads)
    if a.grid == "smoke":
        runs = grid_smoke()
        for x in runs:
            x["threads"], x["device"] = a.threads, "cpu"
        t0 = time.time()
        if a.workers <= 1:
            rows = [run_one(x) for x in runs]
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                rows = list(ex.map(run_one, runs, chunksize=1))
        OUT.mkdir(parents=True, exist_ok=True)
        meta = {"n_runs": len(rows), "wall_s": round(time.time() - t0, 1), "workers": a.workers,
                "threads": a.threads, "torch": torch.__version__, "cpu": True,
                "all_causal_pass": all(r["causal"]["past_future_perturb_maxdiff"] < 1e-6 for r in rows),
                "all_data_label_ok": all(r["data_label_ok"] for r in rows)}
        json.dump({"meta": meta, "rows": rows}, open(OUT / "rate_phase_smoke_summary.json", "w"), indent=1)
        (OUT / "rate_phase_smoke_table.md").write_text(fmt_table(rows) + "\n\n```json\n" + json.dumps(meta) + "\n```\n",
                                                      encoding="utf-8")
        print("\n" + fmt_table(rows))
        print("META", json.dumps(meta))
    elif a.grid == "full":
        runs = grid_full()
        for x in runs:
            x["threads"], x["device"] = a.threads, "cuda"
        rows = []
        for i, x in enumerate(runs):
            print("[%d/%d]" % (i + 1, len(runs)), json.dumps(x), flush=True)
            try:
                rows.append(run_one(x))
            except Exception:
                traceback.print_exc()
            json.dump(rows, open(OUT / "rate_phase_full_partial.json", "w"), indent=1)
        (OUT / "rate_phase_full_table.md").write_text(fmt_table(rows) + "\n", encoding="utf-8")
    else:
        res = run_one(cfg_single)
        print("DONE", json.dumps(res["causal"]))


if __name__ == "__main__":
    main()