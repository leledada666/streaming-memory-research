# -*- coding: utf-8 -*-
"""
Decision-step-aligned injector test (Route A).

At decision step t:
  - only external scalars m_0..m_t are available
  - gated residual bypass injects the NEW scalar for this step
  - model predicts next number m_{t+1}
  - then advance

Prefers RWKV7 HF use_cache recurrent state; falls back to prefix-grow
(feed m_0..m_t length t+1, read last position) if cache fails.
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

ROOT = Path(r"E:\rwkv7")
DEFAULT_MODEL = ROOT / "models" / "RWKV7-Goose-World2.9-0.4B-HF"
DEFAULT_CKPT = ROOT / "runs" / "bypass_nextnum_smoke" / "checkpoint.pt"
DEFAULT_DATA = ROOT / "data" / "series.npy"
DEFAULT_OUT = ROOT / "runs" / "bypass_nextnum_smoke" / "decision_step_test.json"
N_STEPS = 32


class GatedBypassNextNumber(nn.Module):
    """Same pattern as train_bypass.py / one_step_test.py."""

    def __init__(self, backbone: nn.Module, hidden_size: int):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.scalar_enc = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate_logit = nn.Parameter(torch.tensor(0.0))
        self.head = nn.Linear(hidden_size, 1)
        self.probe = nn.Linear(hidden_size, 1)
        self.bos_id = int(getattr(backbone.config, "bos_token_id", 1) or 1)

    def encode_step_embeds(self, m_seq: torch.Tensor):
        B, T = m_seq.shape
        device = m_seq.device
        bos = torch.full((B, T), self.bos_id, dtype=torch.long, device=device)
        base = self.backbone.get_input_embeddings()(bos)
        e = self.scalar_enc(m_seq.unsqueeze(-1))
        gate = torch.sigmoid(self.gate_logit)
        return base + gate * e, gate

    def forward_full(self, m_seq: torch.Tensor):
        inputs_embeds, gate = self.encode_step_embeds(m_seq)
        out = self.backbone.model(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        h = out.last_hidden_state
        pred = self.head(h).squeeze(-1)
        probe = self.probe(h).squeeze(-1)
        return pred, probe, gate


def try_cache_mode(model: GatedBypassNextNumber, m_window: torch.Tensor, n_steps: int):
    past = None
    steps = []
    for t in range(n_steps):
        m_t = m_window[t].view(1, 1)
        inputs_embeds, gate = model.encode_step_embeds(m_t)
        out = model.backbone.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past,
            output_hidden_states=True,
            return_dict=True,
        )
        past = out.past_key_values
        h = out.last_hidden_state[:, -1, :]
        pred = float(model.head(h).squeeze().item())
        probe = float(model.probe(h).squeeze().item())
        inject = float(m_window[t].item())
        true_next = float(m_window[t + 1].item())
        steps.append(
            {
                "t": t,
                "inject_m_t": inject,
                "pred_next": pred,
                "true_next": true_next,
                "abs_err": abs(pred - true_next),
                "probe_cur": probe,
                "probe_abs_err": abs(probe - inject),
            }
        )
    return steps, float(torch.sigmoid(model.gate_logit).item())


def prefix_grow_mode(model: GatedBypassNextNumber, m_window: torch.Tensor, n_steps: int):
    steps = []
    for t in range(n_steps):
        prefix = m_window[: t + 1].view(1, -1)
        inputs_embeds, gate = model.encode_step_embeds(prefix)
        out = model.backbone.model(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        h = out.last_hidden_state[:, -1, :]
        pred = float(model.head(h).squeeze().item())
        probe = float(model.probe(h).squeeze().item())
        inject = float(m_window[t].item())
        true_next = float(m_window[t + 1].item())
        steps.append(
            {
                "t": t,
                "inject_m_t": inject,
                "pred_next": pred,
                "true_next": true_next,
                "abs_err": abs(pred - true_next),
                "probe_cur": probe,
                "probe_abs_err": abs(probe - inject),
            }
        )
    return steps, float(torch.sigmoid(model.gate_logit).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--n-steps", type=int, default=N_STEPS)
    ap.add_argument("--smoke-mae", type=float, default=0.8679395914077759)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    series = np.load(args.data).astype(np.float32)
    seq_len = 64
    start = len(series) - (seq_len + 1) - 10
    n_steps = args.n_steps
    m_np = series[start : start + n_steps + 1]
    m_window = torch.tensor(m_np, dtype=torch.float32, device=device)

    print("loading backbone...", flush=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32
    )
    hidden = int(backbone.config.hidden_size)
    model = GatedBypassNextNumber(backbone, hidden).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.scalar_enc.load_state_dict(ckpt["scalar_enc"])
    model.gate_logit.data.copy_(ckpt["gate_logit"].to(device))
    model.head.load_state_dict(ckpt["head"])
    model.probe.load_state_dict(ckpt["probe"])
    model.eval()

    mode = None
    mode_note = ""
    steps = None
    gate = None
    cache_error = None
    cross = []

    with torch.no_grad():
        try:
            steps, gate = try_cache_mode(model, m_window, n_steps)
            mode = "cache"
            mode_note = (
                "RWKV7 HF use_cache recurrent state worked. "
                "At step t only m_t is newly embedded; past carries m_<t."
            )
        except Exception as e:
            cache_error = f"{type(e).__name__}: {e}"
            print("cache mode failed, falling back to prefix-grow:", flush=True)
            traceback.print_exc()
            steps, gate = prefix_grow_mode(model, m_window, n_steps)
            mode = "prefix-grow"
            mode_note = (
                "Cache unavailable; used prefix-grow: at step t feed m_0..m_t, "
                "read last-position prediction."
            )

        if mode == "cache":
            pg_steps, _ = prefix_grow_mode(model, m_window, min(8, n_steps))
            for a, b in zip(steps[: len(pg_steps)], pg_steps):
                cross.append(
                    {
                        "t": a["t"],
                        "cache_pred": a["pred_next"],
                        "prefix_pred": b["pred_next"],
                        "abs_diff": abs(a["pred_next"] - b["pred_next"]),
                    }
                )

        x_full = m_window[:n_steps].view(1, -1)
        y_full = m_window[1 : n_steps + 1].view(1, -1)
        pred_full, probe_full, _ = model.forward_full(x_full)
        whole_window = {
            "mse_next": float(torch.mean((pred_full - y_full) ** 2).item()),
            "mae_next": float(torch.mean(torch.abs(pred_full - y_full)).item()),
            "last_inject": float(x_full[0, -1].item()),
            "last_pred": float(pred_full[0, -1].item()),
            "last_true": float(y_full[0, -1].item()),
            "last_abs_err": float(torch.abs(pred_full[0, -1] - y_full[0, -1]).item()),
        }

    errs = [s["abs_err"] for s in steps]
    probe_errs = [s["probe_abs_err"] for s in steps]
    mean_ae = float(np.mean(errs))
    result = {
        "design": {
            "route": "A",
            "alignment": "per_decision_step",
            "inject": "one scalar m_t via gated residual bypass into inputs_embeds",
            "task": "predict next number",
            "availability": "at step t only m_<=t visible",
        },
        "ckpt": str(args.ckpt),
        "mode": mode,
        "mode_note": mode_note,
        "cache_error": cache_error,
        "device": str(device),
        "gate": gate,
        "n_steps": n_steps,
        "series_start_index": int(start),
        "mean_abs_err": mean_ae,
        "median_abs_err": float(np.median(errs)),
        "max_abs_err": float(np.max(errs)),
        "mean_probe_abs_err": float(np.mean(probe_errs)),
        "first8_mean_abs_err": float(np.mean(errs[:8])),
        "last8_mean_abs_err": float(np.mean(errs[-8:])),
        "steps": steps,
        "cache_vs_prefix_grow_first8": cross,
        "whole_window_same_span": whole_window,
        "comparison_vs_smoke": {
            "smoke_mean_abs_err": args.smoke_mae,
            "this_mean_abs_err": mean_ae,
            "delta_mae": mean_ae - args.smoke_mae,
            "improved": mean_ae < args.smoke_mae,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "mode": mode,
                "gate": gate,
                "mean_abs_err": mean_ae,
                "median_abs_err": result["median_abs_err"],
                "first8_mae": result["first8_mean_abs_err"],
                "last8_mae": result["last8_mean_abs_err"],
                "vs_smoke_mae": args.smoke_mae,
                "improved": result["comparison_vs_smoke"]["improved"],
                "out": str(out_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
