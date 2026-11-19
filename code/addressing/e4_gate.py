# -*- coding: utf-8 -*-
"""E4 pre-flight gates (CPU).

1. tap-oracle : 1-param tap that reads c at the marker position -> must be 1.0.
2. tap-fixed  : fixed lag=15 tap that ignores the marker -> must be ~0.5.
3. FIR16      : train at the full T/B/steps budget, then bucket by marker lag.
                17-32 must be <=.55 (window leakage check); 0-8 must be high
                (otherwise the gate is inconclusive, not a pass).

Also keeps the future-perturbation + attention-mask causality self-check and
writes everything to e4_gate/e4_gate.json.  Does NOT launch the full run.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import sys

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e4_addressing import build, causal_check, make_batch, nparams, train  # noqa: E402

torch.set_num_threads(4)

# requested buckets: 0-8 / 9-16 / 17-32 / 33-64
RAW_BUCKETS = [(0, 8), (9, 16), (17, 32), (33, 64)]


def buckets_for(T):
    return [(lo, min(hi, T - 1)) for lo, hi in RAW_BUCKETS if lo <= T - 1]


class TapOracle(nn.Module):
    """1-parameter read: logit_t = scale * sum_{j<=t} marker_j * c_j."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, m):
        c, mk = m[..., 0], m[..., 1]
        return self.scale * torch.cumsum(mk * c, dim=1)


class TapFixed(nn.Module):
    """Fixed lag tap: logit_t = scale * c_{t-15}; marker is ignored."""

    def __init__(self, lag=15):
        super().__init__()
        self.lag = lag
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, m):
        B, T, _ = m.shape
        c = m[..., 0]
        out = torch.zeros(B, T, device=m.device, dtype=c.dtype)
        if T > self.lag:
            out[:, self.lag:] = c[:, :T - self.lag]
        return self.scale * out


@torch.no_grad()
def eval_buckets(model, cond, T, device, per_bucket=2048, seed=4242, buckets=None):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    rows = {}
    for lo, hi in (buckets or buckets_for(T)):
        B = per_bucket
        m, y, tq, s = make_batch(B, T, cond, device, gen, lag_range=(lo, hi))
        out = model(m)
        logits = out[0] if isinstance(out, tuple) else out
        pred = torch.where(logits[:, tq[0]] > 0, 1.0, -1.0)
        rows[f"{lo}-{hi}"] = round((pred == y).float().mean().item(), 4)
    # natural uniform-lag batch (same distribution as training)
    B = per_bucket * 2
    m, y, tq, s = make_batch(B, T, cond, device, gen, lag_range=(1, T - 1))
    out = model(m)
    logits = out[0] if isinstance(out, tuple) else out
    pred = torch.where(logits[:, tq[0]] > 0, 1.0, -1.0)
    overall = round((pred == y).float().mean().item(), 4)
    return {"buckets": rows, "overall": overall, "n_per_bucket": per_bucket}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--per_bucket", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(HERE / "e4_gate"))
    a = ap.parse_args()
    T, B, dev = a.T, a.B, a.device
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bs = buckets_for(T)
    t0 = time.time()

    # -- causality self-check (untrained, architecture/mask property) ----------
    causal = {}
    for arm in ("fir16", "gru128", "cross", "bypass"):
        model = build(arm, T, dev).to(dev)
        d = causal_check(model, arm, T=T, device=dev)
        causal[arm] = {"future_perturb_max_logit_diff": d, "pass": bool(d < 1e-5)}
    for name, model in (("tap-oracle", TapOracle()), ("tap-fixed", TapFixed(15))):
        model = model.to(dev)
        d = causal_check(model, name, T=T, device=dev)
        causal[name] = {"params": nparams(model),
                        "future_perturb_max_logit_diff": d,
                        "pass": bool(d < 1e-5)}

    # -- gate 1: tap oracle -----------------------------------------------------
    oracle = TapOracle().to(dev)
    oracle_real = eval_buckets(oracle, "real", T, dev, a.per_bucket, seed=101)
    oracle_sh = eval_buckets(oracle, "shuffle", T, dev, a.per_bucket, seed=102)
    oracle_nm = eval_buckets(oracle, "nomarker", T, dev, a.per_bucket, seed=103)
    oracle_ok = oracle_real["overall"] >= 0.99

    # -- gate 2: fixed lag=15 tap (ignores marker) ------------------------------
    fixed = TapFixed(15).to(dev)
    fixed_real = eval_buckets(fixed, "real", T, dev, a.per_bucket, seed=201)
    fixed_ok = (fixed_real["overall"] <= 0.60 and
                max(fixed_real["buckets"].values()) <= 0.60)

    # -- gate 3: FIR16 at the full training budget ------------------------------
    fir_runs = []
    for seed in seeds:
        model = train("fir16", "real", T, B, a.steps, seed, dev,
                      lr=3e-3, log_every=max(1, a.steps // 4))
        ev = eval_buckets(model, "real", T, dev, a.per_bucket, seed=300 + seed)
        fir_runs.append({"seed": seed, "params": nparams(model), "eval": ev})
        print(json.dumps({"fir16_gate": fir_runs[-1]}), flush=True)
    key_long = f"17-{min(32, T - 1)}"
    key_short = f"0-{min(8, T - 1)}"          # requested bucket is 0-8
    long_vals = [r["eval"]["buckets"].get(key_long, 1.0) for r in fir_runs]
    short_vals = [r["eval"]["buckets"].get(key_short, 0.0) for r in fir_runs]
    fir_long_max = max(long_vals)
    fir_short_mean = sum(short_vals) / len(short_vals)
    fir_long_ok = fir_long_max <= 0.55
    fir_sane = fir_short_mean >= 0.85

    causal_ok = all(v["pass"] for v in causal.values())
    gate_pass = bool(oracle_ok and fixed_ok and fir_long_ok and causal_ok)
    # a broken/undertrained FIR16 can trivially pass the leak check
    gate_status = ("pass" if (gate_pass and fir_sane) else
                   "inconclusive-fir16" if (gate_pass and not fir_sane) else "fail")

    full_cmd = (
        "python e4_addressing.py --T 96 --B 32 --steps 4000 "
        "--arms fir16,gru128,cross,bypass "
        "--conds real,shuffle,nomarker,pos --seeds 0,1,2 "
        "--device cuda --lr 3e-3 --eval_per_bucket 512 "
        "--out e4_full --tag full"
    )
    rep = {
        "config": vars(a),
        "params": {"fir16": nparams(build("fir16", T, dev)),
                   "gru128": nparams(build("gru128", T, dev)),
                   "cross": nparams(build("cross", T, dev)),
                   "bypass": nparams(build("bypass", T, dev)),
                   "tap-oracle": nparams(TapOracle()),
                   "tap-fixed": nparams(TapFixed(15))},
        "causal": causal,
        "gate1_tap_oracle": {"real": oracle_real, "shuffle": oracle_sh,
                             "nomarker": oracle_nm, "pass": bool(oracle_ok)},
        "gate2_tap_fixed15": {"real": fixed_real, "pass": bool(fixed_ok)},
        "gate3_fir16": {"steps": a.steps, "runs": fir_runs,
                        "short_key": key_short, "long_key": key_long,
                        "short_mean": round(fir_short_mean, 4),
                        "long_max": round(fir_long_max, 4),
                        "long_pass": bool(fir_long_ok),
                        "short_sanity_pass": bool(fir_sane)},
        "verdict": {
            "gate_status": gate_status,
            "gate_pass": bool(gate_pass),
            "fir16_sane": bool(fir_sane),
            "causal_pass": bool(causal_ok),
            "reasons": [
                f"oracle overall={oracle_real['overall']:.3f} (need >=.99)",
                f"fixed15 overall={fixed_real['overall']:.3f} (need <=.60)",
                f"fir16 {key_long} max={fir_long_max:.3f} (need <=.55)",
                f"fir16 {key_short} mean={fir_short_mean:.3f} (sanity >=.85)",
                f"causal all pass={causal_ok}",
            ],
            "full_run_command": full_cmd,
            "full_run_launched": False,
            "e4_final_verdict": "pending full T=96 run; gate does not test cross",
        },
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(out / "e4_gate.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)

    print("\n=== E4 GATE (T=%d B=%d steps=%d) ===" % (T, B, a.steps))
    print("tap-oracle real overall", oracle_real["overall"],
          "buckets", oracle_real["buckets"])
    print("tap-oracle shuffle/nomarker", oracle_sh["overall"], oracle_nm["overall"])
    print("tap-fixed15 real overall", fixed_real["overall"],
          "buckets", fixed_real["buckets"])
    for r in fir_runs:
        print("fir16 seed", r["seed"], "buckets", r["eval"]["buckets"],
              "overall", r["eval"]["overall"])
    print("causal", {k: v["pass"] for k, v in causal.items()})
    print("GATE", gate_status, "| pass=", gate_pass, "| fir_sane=", fir_sane)
    print("FULL (not launched):", full_cmd)
    print("DONE", rep["runtime_s"], flush=True)


if __name__ == "__main__":
    main()