# -*- coding: utf-8 -*-
"""Benchmark: streaming state injection (S) vs multi-round re-feed with prefix cache (R) vs
no-cache full recompute (R0, Kronos) vs sliding window (W). INFERENCE ONLY, frozen weights, batch 1.
Measures latency + action-prob consistency (mechanism), not PnL. No orders, no keys.
Parts: --part kronos | rwkv | qwen ; --dtype fp32 | fp16 (fp16 = torch.autocast)."""
from __future__ import annotations
import argparse, json, time, sys, subprocess, contextlib
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

ROOT = Path(r"E:\rwkv7"); sys.path.insert(0, str(ROOT / "kronos_src"))
OUT = ROOT / "runs" / "bench_stream_vs_refeed"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "binance" / "klines_1m" / "bars_1m_of_3m.csv"
ap = argparse.ArgumentParser()
ap.add_argument("--part", required=True); ap.add_argument("--dtype", default="fp32")
ap.add_argument("--reps", type=int, default=3); ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--qwen_rounds", type=int, default=20); ap.add_argument("--reps_rw", type=int, default=1)
ap.add_argument("--cap", type=float, default=300.0); ap.add_argument("--only_SR", type=int, default=0); ap.add_argument("--tag", default=""); ap.add_argument("--mem_cap_gb", type=float, default=6.0)
a = ap.parse_args()
dev = "cuda"; PRIME, STEPS, NS, KX, MAXH, WARM = 64, a.steps, [64, 128, 256], 16, 20, 50
HIST = [100, 500, 1000, 2000]
amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if a.dtype == "fp16" else contextlib.nullcontext
torch.manual_seed(0); T0 = time.time()

def gpu():
    q = "utilization.gpu,temperature.gpu,memory.used,power.draw,clocks.sm"
    s = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    games = subprocess.run(["tasklist", "/FI", "IMAGENAME eq cs2.exe", "/NH"], capture_output=True, text=True).stdout
    return {"nvidia_smi(" + q + ")": s, "cs2_running": "cs2.exe" in games, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
def sync(): torch.cuda.synchronize()
TR = {}
def steps():  # step iterator with a wall-time cap and a VRAM cap (avoid WDDM shared-memory spill); truncation is reported
    t0m = time.perf_counter()
    for t in range(STEPS):
        el = time.perf_counter() - t0m; mem = torch.cuda.memory_reserved() / 1e9
        if el > a.cap or mem > a.mem_cap_gb:
            TR.update({"truncated_at_step": t, "reason": f"time {el:.0f}s > cap" if el > a.cap else f"reserved VRAM {mem:.2f} GB > cap"}); return
        yield t
def lat_stats(lat):
    x = np.array(lat[WARM:]); h = {}
    for k in HIST:
        lo = max(WARM, k - 10); h[str(k)] = float(np.median(lat[lo:k])) if k <= len(lat) else None
    return {"p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)),
            "mean": float(x.mean()), "at_bar": h}
def med_reps(stats):
    out = {k: float(np.median([s[k] for s in stats])) for k in ["p50", "p95", "p99", "mean"]}
    out["at_bar"] = {k: (float(np.median([s["at_bar"][k] for s in stats if s["at_bar"][k] is not None])) if any(s["at_bar"][k] is not None for s in stats) else None) for k in stats[0]["at_bar"]}
    out["per_rep_p95"] = [s["p95"] for s in stats]; return out
def cons(P, Q, legal):
    n = min(len(P), len(Q)); P, Q = P[:n], Q[:n]; P = P[:, legal].astype(np.float64); Q = Q[:, legal].astype(np.float64)
    kl = (P * (np.log(np.clip(P, 1e-30, None)) - np.log(np.clip(Q, 1e-30, None)))).sum(1)
    return {"max_abs": float(np.abs(P - Q).max()), "mean_abs": float(np.abs(P - Q).mean()), "mean_kl": float(kl.mean()),
            "max_kl": float(kl.max()), "n_steps_compared": int(n), "argmax_disagree": float((P.argmax(1) != Q.argmax(1)).mean())}

df = pd.read_csv(DATA); close = df["close"].to_numpy(np.float64); N = len(close)
logret = np.concatenate([[0.0], np.diff(np.log(close))]); r_bps = (logret * 1e4).astype(np.float32)
mom = np.array([r_bps[max(0, t - 4):t + 1].sum() for t in range(N)], np.float32)
i1, i2 = int(N * 0.70), int(N * 0.85)
s0 = i2 + PRIME; bars = np.arange(s0 - PRIME, s0 + STEPS); L = len(bars)  # warmup = first 64 test bars
res = {"part": a.part, "dtype": a.dtype, "reps": a.reps, "steps": STEPS, "prime": PRIME, "slice_bars": [int(bars[0]), int(bars[-1])],
       "split": [i1, i2 - i1, N - i2], "gpu_start": gpu(), "modes": {}}
print(json.dumps(res["gpu_start"]), flush=True)
LEGAL = np.array([True, True, True, False])  # pos=0 always (no execution) -> close illegal
lmask = torch.tensor(LEGAL, device=dev)
def to_probs(lg): return torch.softmax(lg.float().masked_fill(~lmask, -1e9), -1)

def run_mode(name, fn, reps=None):
    stats, Ps, trs = [], [], []
    for r in range(reps or a.reps):
        torch.cuda.empty_cache(); sync(); TR.clear()
        lat, P, extra = fn(); trs.append(dict(TR)); extra = dict(extra or {}); extra["n_steps_done"] = len(lat)
        stats.append(lat_stats(lat)); Ps.append(np.stack(P)); print(name, "rep", r, json.dumps(stats[-1]), flush=True)
    nmin = min(len(p) for p in Ps); Ps = [p[:nmin] for p in Ps]
    rec = {"latency_ms": med_reps(stats), "rep_prob_maxdiff": float(max(np.abs(Ps[0] - p).max() for p in Ps)), "n_reps": len(stats), "truncation": trs}
    if extra: rec["extra"] = extra
    res["modes"][name] = rec; return Ps[0]

# =====================================================================================
if a.part == "kronos":
    from model import Kronos, KronosTokenizer
    _o, _h, _l, _c = (df[k].to_numpy(np.float64) for k in ["open", "high", "low", "close"])
    _raw = {"r": r_bps.astype(np.float64), "mom": mom.astype(np.float64),
            "logvol": np.log(np.maximum(df["base_vol"].to_numpy(np.float64), 1e-9)),
            "wick": ((_h - np.maximum(_o, _c)) - (np.minimum(_o, _c) - _l)) / _c * 1e4,
            "range": np.log(np.maximum((_h - _l) / _c * 1e4, 1e-3)), "imbalance": df["imbalance"].to_numpy(np.float64)}
    def _tz(x):
        sr = pd.Series(x); mu = sr.shift(1).rolling(240, min_periods=30).mean(); sd = sr.shift(1).rolling(240, min_periods=30).std()
        return np.clip(((sr - mu) / (sd + 1e-9)).fillna(0).to_numpy(), -5, 5).astype(np.float32)
    MKT = np.stack([_tz(_raw[k]) for k in ["r", "mom", "logvol", "wick", "range", "imbalance"]], 1)
    feat = np.stack([df["open"], df["high"], df["low"], df["close"], df["base_vol"], df["quote_vol"]], 1).astype(np.float64)
    mu, sd = feat[:i1].mean(0), feat[:i1].std(0); xn = np.clip((feat - mu) / (sd + 1e-5), -5, 5).astype(np.float32)
    tok = KronosTokenizer.from_pretrained(str(ROOT / "models" / "Kronos-Tokenizer-base")).to(dev).eval()
    model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
    for p in list(tok.parameters()) + list(model.parameters()): p.requires_grad = False
    xt = torch.tensor(xn, device=dev); S1 = np.zeros(L, np.int64); S2 = np.zeros(L, np.int64)
    with torch.no_grad():
        for b0 in range(0, L, 256):
            idx = [list(range(t - 63, t + 1)) for t in bars[b0:b0 + 256]]
            z = tok.encode(xt[torch.tensor(idx, device=dev)], half=True)
            S1[b0:b0 + len(idx)] = z[0][:, -1].cpu().numpy(); S2[b0:b0 + len(idx)] = z[1][:, -1].cpu().numpy()
    ts = pd.to_datetime(df["ts"], unit="s", utc=True)
    STAMP = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1).astype(np.float32)[bars]
    S1t, S2t, STt = (torch.tensor(x, device=dev)[None] for x in (S1, S2, STAMP))
    H = model.d_model
    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.enc = nn.Sequential(nn.Linear(8, H), nn.GELU(), nn.Linear(H, H))
            s.enc_head = nn.Sequential(nn.Linear(8, H), nn.GELU(), nn.Linear(H, H))
            s.gate_logit = nn.Parameter(torch.tensor(-2.0))
            s.pi = nn.Sequential(nn.Linear(2 * H, 128), nn.GELU(), nn.Linear(128, 4))
            s.v = nn.Sequential(nn.Linear(2 * H, 128), nn.GELU(), nn.Linear(128, 1))
        def delta(s, m):
            u = s.enc(m); u = u / (u.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6); return torch.sigmoid(s.gate_logit) * u
        def heads(s, h, m):
            z = torch.cat([h / (h.norm(dim=-1, keepdim=True) + 1e-6) * 8, s.enc_head(m)], -1); return s.pi(z)
    net = Net().to(dev).eval(); ck = ROOT / "runs" / "kronos_rl_1m_direct_bn3m_pertrade" / "seed0" / "best.pt"
    if ck.exists(): net.load_state_dict(torch.load(ck)); res["head"] = str(ck)
    else: res["head"] = "random seed0"
    M = torch.tensor(np.concatenate([MKT[bars], np.zeros((L, 2), np.float32)], 1), device=dev)[None]
    fh = model.forward_hidden
    res["R_definition"] = (f"per round: one cached forward of [new bar (bypass on) + {KX} fixed extra positions (s1=1..{KX}, s2=0, "
                           f"stamp of the bar, no bypass)], DIRECT-head readout at the last extra position (m = current bar), argmax; then "
                           f"one cached forward of 1 write-back position encoding the action (s1=100+action, s2=0, no bypass). "
                           f"Context grows by {KX + 2} positions per bar. Kronos has no text, so this emulates prompt+output overhead.")
    res["R0_definition"] = "no cache: full causal forward over all bars [0..t] every step, readout at last position"
    res["W_definition"] = "no reusable cache (window start shifts): full forward over the last N bars every step, readout last"
    res["reference"] = "one full causal forward (no cache) over all 2064 positions, same bypass content; readout at each decision position"
    with torch.no_grad(), amp():
        h = fh(S1t, S2t, stamp=STt, bypass_delta=net.delta(M)); P_ref = to_probs(net.heads(h[0, PRIME:], M[0, PRIME:])).cpu().numpy()
        del h
        def mS():
            lat, P = [], []
            _, past = fh(S1t[:, :PRIME], S2t[:, :PRIME], stamp=STt[:, :PRIME], use_cache=True, bypass_delta=net.delta(M[:, :PRIME]))
            for t in steps():
                j = PRIME + t; sync(); t1 = time.perf_counter()
                h, past = fh(S1t[:, j:j + 1], S2t[:, j:j + 1], stamp=STt[:, j:j + 1], past_key_values=past, use_cache=True,
                             bypass_delta=net.delta(M[:, j:j + 1]))
                p = to_probs(net.heads(h[0, -1], M[0, j])); act = int(p.argmax()); sync(); lat.append((time.perf_counter() - t1) * 1e3)
                P.append(p.cpu().numpy())
            return lat, P, None
        XS1 = torch.arange(1, KX + 1, device=dev)[None]; XS2 = torch.zeros_like(XS1); ZD = torch.zeros(1, KX, H, device=dev)
        def mR():
            lat, P = [], []
            _, past = fh(S1t[:, :PRIME], S2t[:, :PRIME], stamp=STt[:, :PRIME], use_cache=True, bypass_delta=net.delta(M[:, :PRIME]))
            for t in steps():
                j = PRIME + t; sync(); t1 = time.perf_counter()
                s1 = torch.cat([S1t[:, j:j + 1], XS1], 1); s2 = torch.cat([S2t[:, j:j + 1], XS2], 1)
                st = STt[:, j:j + 1].expand(1, KX + 1, 5)
                h, past = fh(s1, s2, stamp=st, past_key_values=past, use_cache=True,
                             bypass_delta=torch.cat([net.delta(M[:, j:j + 1]), ZD.to(net.delta(M[:, j:j + 1]).dtype)], 1))
                p = to_probs(net.heads(h[0, -1], M[0, j])); act = int(p.argmax())
                _, past = fh(torch.tensor([[100 + act]], device=dev), torch.zeros(1, 1, dtype=torch.long, device=dev),
                             stamp=STt[:, j:j + 1], past_key_values=past, use_cache=True)
                sync(); lat.append((time.perf_counter() - t1) * 1e3); P.append(p.cpu().numpy())
            return lat, P, {"final_context_positions": PRIME + STEPS * (KX + 2)}
        def mR0():
            lat, P = [], []
            for t in steps():
                j = PRIME + t; sync(); t1 = time.perf_counter()
                h = fh(S1t[:, :j + 1], S2t[:, :j + 1], stamp=STt[:, :j + 1], bypass_delta=net.delta(M[:, :j + 1]))
                p = to_probs(net.heads(h[0, -1], M[0, j])); act = int(p.argmax()); sync(); lat.append((time.perf_counter() - t1) * 1e3)
                P.append(p.cpu().numpy())
            return lat, P, None
        def mW(Nw):
            def f():
                lat, P = [], []
                for t in steps():
                    j = PRIME + t; lo = max(0, j - Nw + 1); sync(); t1 = time.perf_counter()
                    h = fh(S1t[:, lo:j + 1], S2t[:, lo:j + 1], stamp=STt[:, lo:j + 1], bypass_delta=net.delta(M[:, lo:j + 1]))
                    p = to_probs(net.heads(h[0, -1], M[0, j])); act = int(p.argmax()); sync(); lat.append((time.perf_counter() - t1) * 1e3)
                    P.append(p.cpu().numpy())
                return lat, P, None
            return f
        PS = run_mode("S", mS); res["modes"]["S"]["cons_vs_ref"] = cons(P_ref, PS, LEGAL)
        PR = run_mode("R", mR, a.reps_rw); res["modes"]["R"]["cons_vs_S"] = cons(PS, PR, LEGAL)
        if not a.only_SR:
            P0 = run_mode("R0", mR0); res["modes"]["R0"]["cons_vs_ref"] = cons(P_ref, P0, LEGAL)
            for Nw in NS:
                PW = run_mode(f"W{Nw}", mW(Nw), a.reps_rw); res["modes"][f"W{Nw}"]["cons_vs_S"] = cons(PS, PW, LEGAL)
    res["ref_action_freq"] = np.bincount(P_ref.argmax(1), minlength=4).tolist()

# =====================================================================================
elif a.part == "rwkv":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL = ROOT / "models" / "RWKV7-Goose-World2.9-0.4B-HF"
    bb = AutoModelForCausalLM.from_pretrained(str(MODEL), trust_remote_code=True, torch_dtype=torch.float32).to(dev).eval()
    for p in bb.parameters(): p.requires_grad = False
    rtok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    H = int(bb.config.hidden_size); BOS = int(getattr(bb.config, "bos_token_id", 1) or 1); EMB = bb.get_input_embeddings()
    class RNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.enc = nn.Sequential(nn.Linear(4, H), nn.GELU(), nn.Linear(H, H)); s.gate_logit = nn.Parameter(torch.tensor(0.0))
            s.pi = nn.Sequential(nn.Linear(H, 128), nn.GELU(), nn.Linear(128, 4)); s.v = nn.Sequential(nn.Linear(H, 128), nn.GELU(), nn.Linear(128, 1))
        def embeds(s, m):
            return EMB(torch.full(m.shape[:2], BOS, dtype=torch.long, device=m.device)) + torch.sigmoid(s.gate_logit) * s.enc(m)
        def heads(s, h): return s.pi(h / (h.norm(dim=-1, keepdim=True) + 1e-6) * 8)
    net = RNet().to(dev).eval(); ck = ROOT / "runs" / "rl_kline_1m_pnl" / "best.pt"
    if ck.exists(): net.load_state_dict(torch.load(ck)); res["head"] = str(ck)
    else: res["head"] = "random seed0"
    r_std = float(r_bps[:i1].std()); m_std = float(mom[:i1].std())
    Mr = torch.tensor(np.stack([r_bps[bars] / r_std, mom[bars] / m_std, np.zeros(L), np.zeros(L)], 1).astype(np.float32), device=dev)[None]
    PROMPT = "\n\nUser: A new 1-minute bar was injected. Decide the next action: long, short, hold or close.\n\nAssistant:"
    pid = torch.tensor([rtok.encode(PROMPT)], device=dev); NOUT = 3
    res["R_definition"] = (f"per round: one cached forward of [new bar embedding (BOS emb + bypass) + {pid.shape[1]} real text tokens of a fixed "
                           f"decision prompt {PROMPT!r}], policy-head readout at the last prompt token, argmax; then greedy-decode {NOUT} text "
                           f"tokens with the LM head, each fed back through the cached recurrent state (so the state carries bar+prompt+output). "
                           f"{NOUT + 1} model calls per round; {1 + pid.shape[1] + NOUT} new positions per bar.")
    res["W_definition"] = "recurrent state cannot drop old bars: re-run from zero state over the last N bars every step (no cache), readout last"
    res["reference"] = "one full forward (no cache, FLA chunk kernel since len>=64) over all 2064 bar embeddings; readout at each decision position"
    res["kernel_note"] = "FLA RWKV7: seq_len>=64 -> chunk_rwkv7, else fused recurrent. S steps use recurrent; reference/W/prime use chunk."
    def fwd(**kw): return bb.model(return_dict=True, **kw)
    with torch.no_grad(), amp():
        P_ref = to_probs(net.heads(fwd(inputs_embeds=net.embeds(Mr), use_cache=False).last_hidden_state[0, PRIME:])).cpu().numpy()
        def mS():
            lat, P = [], []
            past = fwd(inputs_embeds=net.embeds(Mr[:, :PRIME]), use_cache=True).past_key_values
            for t in steps():
                j = PRIME + t; sync(); t1 = time.perf_counter()
                o = fwd(inputs_embeds=net.embeds(Mr[:, j:j + 1]), past_key_values=past, use_cache=True); past = o.past_key_values
                p = to_probs(net.heads(o.last_hidden_state[0, -1])); act = int(p.argmax()); sync(); lat.append((time.perf_counter() - t1) * 1e3)
                P.append(p.cpu().numpy())
            return lat, P, None
        gen_texts = []
        def mR():
            lat, P = [], []; gen_texts.clear()
            past = fwd(inputs_embeds=net.embeds(Mr[:, :PRIME]), use_cache=True).past_key_values
            for t in steps():
                j = PRIME + t; sync(); t1 = time.perf_counter()
                x = torch.cat([net.embeds(Mr[:, j:j + 1]), EMB(pid).to(net.embeds(Mr[:, j:j + 1]).dtype)], 1)
                o = fwd(inputs_embeds=x, past_key_values=past, use_cache=True); past = o.past_key_values
                hl = o.last_hidden_state[:, -1]; p = to_probs(net.heads(hl[0])); act = int(p.argmax())
                nt = bb.lm_head(hl).argmax(-1); toks = []
                for k in range(NOUT):
                    toks.append(nt); o = fwd(input_ids=nt[:, None], past_key_values=past, use_cache=True); past = o.past_key_values
                    if k < NOUT - 1: nt = bb.lm_head(o.last_hidden_state[:, -1]).argmax(-1)
                sync(); lat.append((time.perf_counter() - t1) * 1e3); P.append(p.cpu().numpy())
                if t < 5 or t == STEPS - 1: gen_texts.append(rtok.decode(torch.cat(toks).tolist()))
            return lat, P, {"sample_generated_text": list(gen_texts), "prompt_tokens": int(pid.shape[1]), "out_tokens": NOUT}
        def mW(Nw):
            def f():
                lat, P = [], []
                for t in steps():
                    j = PRIME + t; lo = max(0, j - Nw + 1); sync(); t1 = time.perf_counter()
                    o = fwd(inputs_embeds=net.embeds(Mr[:, lo:j + 1]), use_cache=False)
                    p = to_probs(net.heads(o.last_hidden_state[0, -1])); act = int(p.argmax()); sync(); lat.append((time.perf_counter() - t1) * 1e3)
                    P.append(p.cpu().numpy())
                return lat, P, None
            return f
        PS = run_mode("S", mS); res["modes"]["S"]["cons_vs_ref"] = cons(P_ref, PS, LEGAL)
        PR = run_mode("R", mR, a.reps_rw); res["modes"]["R"]["cons_vs_S"] = cons(PS, PR, LEGAL)
        for Nw in NS:
            PW = run_mode(f"W{Nw}", mW(Nw), a.reps_rw); res["modes"][f"W{Nw}"]["cons_vs_S"] = cons(PS, PW, LEGAL)
    res["ref_action_freq"] = np.bincount(P_ref.argmax(1), minlength=4).tolist()

# =====================================================================================
elif a.part == "qwen":
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    QM = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
    qt = AutoTokenizer.from_pretrained(str(QM)); qm = AutoModelForCausalLM.from_pretrained(str(QM), torch_dtype=torch.float16).to(dev).eval()
    EOS = {qt.convert_tokens_to_ids("<|im_end|>"), qt.eos_token_id}; MAXNEW = 64
    ts = pd.to_datetime(df["ts"], unit="s", utc=True)
    def line(t): return f"{ts[t].strftime('%H:%M')} o={df['open'][t]:.1f} h={df['high'][t]:.1f} l={df['low'][t]:.1f} c={df['close'][t]:.1f} v={df['base_vol'][t]:.1f} imb={df['imbalance'][t]:+.2f}"
    SYS = ("You are a BTC/USDT perpetual scalp trader. Each turn you see 1-minute bars (time, open, high, low, close, volume, "
           "taker imbalance). Reply with exactly one of LONG, SHORT, HOLD, CLOSE, then one short reason.")
    def fresh_ids(b):
        txt = "Last 60 bars:\n" + "\n".join(line(t) for t in range(b - 59, b + 1)) + "\nDecision?"
        return qt.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": txt}], add_generation_prompt=True, return_tensors="pt").to(dev)
    @torch.no_grad()
    def gen(ids, cache):
        sync(); t1 = time.perf_counter()
        o = qm(input_ids=ids, past_key_values=cache, use_cache=True); cache = o.past_key_values
        nt = o.logits[:, -1].argmax(-1); sync(); ttft = (time.perf_counter() - t1) * 1e3; out = [int(nt)]
        while len(out) < MAXNEW and out[-1] not in EOS:
            o = qm(input_ids=nt[:, None], past_key_values=cache, use_cache=True); cache = o.past_key_values; nt = o.logits[:, -1].argmax(-1); out.append(int(nt))
        sync(); return ttft, (time.perf_counter() - t1) * 1e3, out, cache
    R = a.qwen_rounds; recs = {"fresh": [], "cached": []}; texts = {}
    for r in range(R):
        ids = fresh_ids(s0 + r); ttft, tot, out, _ = gen(ids, DynamicCache())
        recs["fresh"].append({"ttft": ttft, "total": tot, "new": len(out), "prompt": int(ids.shape[1])})
        if r < 2: texts.setdefault("fresh", []).append(qt.decode(out))
    cache = DynamicCache(); ids = fresh_ids(s0); ctx = 0
    for r in range(R):
        if r > 0:
            msg = f"<|im_start|>user\nNew bar: {line(s0 + r)}\nDecision?<|im_end|>\n<|im_start|>assistant\n"
            pre = "" if last in EOS else "<|im_end|>"
            ids = torch.tensor([[last] + qt.encode(pre + "\n" + msg)], device=dev)
        ttft, tot, out, cache = gen(ids, cache); last = out[-1]; ctx += int(ids.shape[1]) + len(out) - 1
        recs["cached"].append({"ttft": ttft, "total": tot, "new": len(out), "input": int(ids.shape[1]), "ctx_after": ctx})
        if r < 2 or r == R - 1: texts.setdefault("cached", []).append(qt.decode(out))
    def summ(v, k):
        x = np.array([q[k] for q in v[2:]]); return {"p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)), "max": float(x.max())}
    res["qwen"] = {m: {"ttft_ms": summ(v, "ttft"), "total_ms": summ(v, "total"), "mean_new_tokens": float(np.mean([q["new"] for q in v])),
                       "rounds": R, "discarded_warmup_rounds": 2, "per_round": v} for m, v in recs.items()}
    res["qwen"]["sample_text"] = texts
    res["qwen"]["definition"] = ("fp16, greedy, cap 64 new tokens, stop at <|im_end|>. fresh: system+user prompt with last 60 bars as text, "
                                 "new DynamicCache each round. cached: round0 = same prompt; each later round appends only the new bar as a user "
                                 "turn on top of the carried KV cache (which includes all previous replies).")
res["gpu_end"] = gpu(); res["wall_s"] = round(time.time() - T0, 1)
json.dump(res, open(OUT / f"results_{a.part}_{a.dtype}{a.tag}.json", "w"), indent=1)
print("DONE", a.part, a.dtype, res["wall_s"], json.dumps(res["gpu_end"]), flush=True)