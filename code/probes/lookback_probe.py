# -*- coding: utf-8 -*-
"""Read-only attention look-back probe. No training, no weight updates.

Two substrates, identical statistics:
  A) Kronos-small (8 layers, 8 heads, d_model=512) on its native financial token stream.
  B) Qwen2.5-1.5B-Instruct (28 layers x 12 q-heads / 2 kv-heads, GQA 6:1) on real text.
     Only the first K decoder layers are instantiated (memory), attention stats are exact for those.

Statistic per (layer, head):
  mass_at_lag L for query t  = A[t, t-L]
  ratio(L)                   = mean_t A[t, t-L] / mean_t 1/(t+1)     (uniform-over-prefix baseline)
  t ranges over max(L, T//4) .. T-1  (drop shortest prefixes, avoid baseline blow-up)
Also: last-4-key mass, normalised entropy, per-head max ratio.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(r"E:\rwkv7")
LAGS = [1, 2, 5, 10, 15, 30, 60]


# ------------------------------------------------------------------ statistics
def stats_from_probs(probs, lags=LAGS, prefix_frac=0.25):
    """probs: float32 tensor (B,H,T,T) already causal-masked. Returns dict."""
    B, H, T, _ = probs.shape
    p = probs.double()
    out = {"T": T, "H": H, "ratio": {}, "ratio_headmax": {}, "per_layer_L10": []}
    tmin = T // 4
    for L in lags:
        lo = max(L, tmin)
        if lo >= T:
            out["ratio"][L] = float("nan"); out["ratio_headmax"][L] = float("nan"); continue
        idx = torch.arange(lo, T)
        mass = p[:, :, idx, idx - L]                     # (B,H,n)
        num = mass.mean(dim=(0, 2))                      # (H,)
        base = torch.tensor([1.0 / (t + 1) for t in idx.tolist()]).mean().double()
        out["ratio"][L] = float((num.mean() / base).item())
        out["ratio_headmax"][L] = float((num / base).max().item())
    # last-4 mass
    lo = max(3, tmin)
    idx = torch.arange(lo, T)
    last4 = sum(p[:, :, idx, idx - d] for d in range(4))
    out["last4_mass"] = float(last4.mean().item())
    # normalised entropy
    ent = -(p.clamp_min(1e-12).log() * p).sum(-1)        # (B,H,T)
    lo2 = max(8, tmin)
    idx2 = torch.arange(lo2, T)
    norm = torch.tensor([math.log(t + 1) for t in idx2.tolist()]).double()
    out["entropy_norm"] = float((ent[:, :, idx2] / norm).mean().item())
    out["entropy_abs"] = float(ent[:, :, idx2].mean().item())
    return out


def per_layer_L10(probs, L=10):
    B, H, T, _ = probs.shape
    lo = max(L, T // 4)
    idx = torch.arange(lo, T)
    base = torch.tensor([1.0 / (t + 1) for t in idx.tolist()]).mean().double()
    mass = probs.double()[:, :, idx, idx - L].mean(dim=(0, 2))
    return (mass / base).tolist()                         # per (q) head


def merge_per_layer(per_layer_list):
    """per_layer_list: list over layers of list-over-heads ratios -> (L, H) means."""
    return np.array(per_layer_list, dtype=np.float64)     # (layers, heads)


# ------------------------------------------------------------------ Kronos
def run_kronos(args):
    sys.path.insert(0, str(ROOT / "kronos_src"))
    from model import Kronos, KronosTokenizer  # noqa: F401
    import pandas as pd

    dev = "cpu"
    model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = len(model.transformer)
    n_heads = model.transformer[0].self_attn.n_heads

    store = {}
    for i, layer in enumerate(model.transformer):
        mod = layer.self_attn
        orig = mod.forward

        def make_fwd(mod=mod, orig=orig, i=i):
            def fwd(x, **kw):
                if kw.get("past_kv") is None and store.get(i) is None:
                    with torch.no_grad():
                        b, T, _ = x.shape
                        q = mod.q_proj(x).view(b, T, mod.n_heads, mod.head_dim).transpose(1, 2)
                        k = mod.k_proj(x).view(b, T, mod.n_heads, mod.head_dim).transpose(1, 2)
                        q, k = mod.rotary(q, k, position_offset=0)
                        s = (q @ k.transpose(-1, -2)) / math.sqrt(mod.head_dim)
                        m = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
                        s = s.masked_fill(m, float("-inf"))
                        store[i] = torch.softmax(s, -1).float()
                return orig(x, **kw)
            return fwd

        mod.forward = make_fwd()

    z = np.load(ROOT / "runs" / "e6_long_2026-10-20" / "e5_tokens.npz")
    S1, S2 = z["S1"], z["S2"]
    DATA = ROOT / "data" / "binance" / "klines_1m" / "bars_1m_of_3m.csv"
    df = pd.read_csv(DATA, usecols=["ts"])
    ts = pd.to_datetime(df["ts"], unit="s", utc=True)
    STAMP = np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], 1).astype(np.float32)
    N = min(len(S1), len(STAMP))

    T = args.T
    B = args.bs
    starts = np.linspace(int(N * 0.7), N - T - 1, B).astype(int)
    results = []
    per_layer_all = []
    ctrl_layers = []
    for T_ in (T,):
        s1 = torch.tensor(np.stack([S1[s:s + T_] for s in starts]))
        s2 = torch.tensor(np.stack([S2[s:s + T_] for s in starts]))
        st = torch.tensor(np.stack([STAMP[s:s + T_] for s in starts]))
        if args.control:
            g = torch.Generator().manual_seed(0)
            s1 = torch.randint(0, 1024, (B, T_), generator=g)
            s2 = torch.randint(0, 1024, (B, T_), generator=g)
            st = torch.zeros_like(st)
        store.clear()
        with torch.no_grad():
            model.forward_hidden(s1, s2, stamp=st)
        probs = torch.stack([store[i] for i in range(n_layers)])   # (L,B,H,T,T)
        st_layer = []
        for li in range(n_layers):
            d = stats_from_probs(probs[li])
            d["layer"] = li
            d["head_mean_L10"] = float(np.mean(per_layer_L10(probs[li])))
            st_layer.append(d)
            per_layer_all.append(per_layer_L10(probs[li]))
        results.append({"T": T_, "layers": st_layer, "n_layers": n_layers, "n_heads": n_heads})
    return {
        "model": "Kronos-small" + (" [random-token control]" if args.control else ""),
        "params": int(sum(p.numel() for p in model.parameters())),
        "n_layers": n_layers, "n_heads": n_heads, "gqa": False,
        "runs": results, "per_layer_L10": np.array(per_layer_all).tolist(),
        "substrate": "native financial tokens (S1/S2 + temporal stamp)",
    }


# ------------------------------------------------------------------ Qwen
def build_text(min_tokens, tokenizer):
    files = [ROOT / "kronos_src" / "README.md", ROOT / "kronos_src" / "finetune_csv" / "README.md",
             ROOT / "kronos_src" / "finetune" / "README.md", ROOT / "README.md"]
    txt = []
    for f in files:
        if f.exists():
            txt.append(f.read_text(encoding="utf-8", errors="ignore"))
    for f in sorted(ROOT.glob("*.md")):
        txt.append(f.read_text(encoding="utf-8", errors="ignore"))
    text = "\n\n".join(txt)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    return ids, len(ids)


def run_qwen(args):
    from transformers import AutoTokenizer, Qwen2Config, Qwen2Model

    path = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(str(path))
    cfg = Qwen2Config.from_pretrained(str(path))
    K = args.qwen_layers
    full_layers = cfg.num_hidden_layers
    cfg.num_hidden_layers = K
    cfg.use_sliding_window = False
    cfg.sliding_window = None
    cfg._attn_implementation = "eager"
    torch.set_grad_enabled(False)
    # meta-init + selective shard load => peak RSS ~= loaded slice only (no fp32 copy, no lm_head)
    model = Qwen2Model.from_pretrained(
        str(path), config=cfg, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation="eager",
        ignore_mismatched_sizes=True,
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    text_ids, ntok = build_text(0, tok)
    T = args.T
    B = args.bs
    span = ntok - T - 1
    starts = np.linspace(0, span, B).astype(int)
    inputs = torch.stack([text_ids[s:s + T] for s in starts])

    mask = torch.ones_like(inputs)
    with torch.no_grad():
        out = model(input_ids=inputs, attention_mask=mask, output_attentions=True, use_cache=False)
    probs_all = torch.stack([a.float() for a in out.attentions])   # (K,B,12,T,T)

    per_layer_all = []
    st_layer = []
    for li in range(K):
        d = stats_from_probs(probs_all[li])
        d["layer"] = li
        st_layer.append(d)
        per_layer_all.append(per_layer_L10(probs_all[li]))

    # control: random tokens
    g = torch.Generator().manual_seed(0)
    rnd = torch.randint(0, cfg.vocab_size, (B, T), generator=g)
    with torch.no_grad():
        out2 = model(input_ids=rnd, attention_mask=torch.ones_like(rnd),
                     output_attentions=True, use_cache=False)
    ctrl = []
    for li in range(K):
        ctrl.append(stats_from_probs(out2.attentions[li].float()))

    return {
        "model": f"Qwen2.5-1.5B-Instruct (first {K}/{full_layers} layers)",
        "params_loaded": int(sum(p.numel() for p in model.parameters())),
        "n_layers_loaded": K, "n_heads": cfg.num_attention_heads,
        "n_kv_heads": cfg.num_key_value_heads, "gqa": True,
        "runs": [{"T": T, "layers": st_layer}],
        "per_layer_L10": np.array(per_layer_all).tolist(),
        "control_random_tokens": {"T": T, "layers": ctrl,
                                  "ratio": [c["ratio"] for c in ctrl]},
        "substrate": "real English/technical text (repo READMEs), contiguous windows",
        "text_tokens": int(ntok),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["kronos", "qwen"], required=True)
    ap.add_argument("--T", type=int, default=192)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--qwen_layers", type=int, default=12)
    ap.add_argument("--control", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(6)
    res = run_kronos(a) if a.which == "kronos" else run_qwen(a)
    res["args"] = vars(a)
    if a.out:
        Path(a.out).write_text(json.dumps(res), encoding="utf-8")
    print(json.dumps(res))


if __name__ == "__main__":
    main()