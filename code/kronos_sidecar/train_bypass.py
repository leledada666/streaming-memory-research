# -*- coding: utf-8 -*-
"""
Minimal RWKV7 + gated-residual bypass toy trainer.

Route A toy: each decision step injects one external scalar via bypass into
inputs_embeds (token emb + gate * encode(m_t)). Task: predict next number.

Feasibility run notes:
- gate_logit init raised from -2.0 (~0.12) so bypass can open; optional
  temporary L2 nudge toward a target gate early in training.
- Backbone stays frozen unless --unfreeze-backbone.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM


ROOT = Path(r"E:\rwkv7")
DEFAULT_MODEL = ROOT / "models" / "RWKV7-Goose-World2.9-0.4B-HF"
DEFAULT_DATA = ROOT / "data"
DEFAULT_RUNS = ROOT / "runs"


class SeriesWindowDataset(Dataset):
    def __init__(self, series: np.ndarray, seq_len: int, stride: int = 1):
        self.series = torch.tensor(series, dtype=torch.float32)
        self.seq_len = seq_len
        self.max_start = len(series) - (seq_len + 1)
        if self.max_start < 0:
            raise ValueError("series too short for seq_len")
        self.starts = list(range(0, self.max_start + 1, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        window = self.series[s : s + self.seq_len + 1]
        x = window[:-1]
        y = window[1:]
        return x, y


class GatedBypassNextNumber(nn.Module):
    """RWKV7 backbone + gated residual bypass + next-number head (+ probe)."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        freeze_backbone: bool = True,
        gate_logit_init: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.scalar_enc = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # Default 0.0 -> sigmoid ~0.5 (smoke used -2.0 -> ~0.12 and stayed stuck)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_logit_init)))
        self.head = nn.Linear(hidden_size, 1)
        self.probe = nn.Linear(hidden_size, 1)
        self.bos_id = int(getattr(backbone.config, "bos_token_id", 1) or 1)

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def forward(self, m_seq: torch.Tensor):
        B, T = m_seq.shape
        device = m_seq.device
        bos = torch.full((B, T), self.bos_id, dtype=torch.long, device=device)
        base = self.backbone.get_input_embeddings()(bos)
        e = self.scalar_enc(m_seq.unsqueeze(-1))
        gate = torch.sigmoid(self.gate_logit)
        inputs_embeds = base + gate * e

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


def evaluate(model, loader, device, probe_w: float):
    model.eval()
    total_main, total_probe, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred, probe, _ = model(x)
            main = torch.mean((pred - y) ** 2).item()
            pr = torch.mean((probe - x) ** 2).item()
            bs = x.size(0)
            total_main += main * bs
            total_probe += pr * bs
            n += bs
    return total_main / max(n, 1), total_probe / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--out", type=str, default=str(DEFAULT_RUNS / "bypass_nextnum_smoke"))
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--probe-weight", type=float, default=0.1)
    ap.add_argument("--freeze-backbone", action="store_true", default=True)
    ap.add_argument("--unfreeze-backbone", action="store_true")
    ap.add_argument(
        "--gate-logit-init",
        type=float,
        default=0.0,
        help="Initial gate_logit; sigmoid(0)=0.5. Smoke used -2.0 (~0.12).",
    )
    ap.add_argument(
        "--gate-nudge-weight",
        type=float,
        default=0.0,
        help="Optional L2 nudge weight on (gate - gate_target). 0 disables.",
    )
    ap.add_argument(
        "--gate-target",
        type=float,
        default=0.5,
        help="Target gate value for optional nudge (sigmoid space).",
    )
    ap.add_argument(
        "--gate-nudge-steps",
        type=int,
        default=100,
        help="Apply gate nudge only for the first N steps (then weight=0).",
    )
    args = ap.parse_args()

    freeze = not args.unfreeze_backbone
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data)
    series = np.load(data_dir / "series.npy")
    n = len(series)
    split = int(n * 0.9)
    train_series = series[:split]
    val_series = series[split - args.seq_len :]

    train_ds = SeriesWindowDataset(train_series, args.seq_len, stride=8)
    val_ds = SeriesWindowDataset(val_series, args.seq_len, stride=args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} cuda={torch.cuda.is_available()}")
    if device.type == "cuda":
        print("gpu=", torch.cuda.get_device_name(0))

    print("loading backbone from", args.model)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    hidden = int(backbone.config.hidden_size)
    model = GatedBypassNextNumber(
        backbone,
        hidden,
        freeze_backbone=freeze,
        gate_logit_init=args.gate_logit_init,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    print(
        "trainable params:",
        sum(p.numel() for p in params),
        "freeze_backbone=",
        freeze,
        "gate_logit_init=",
        args.gate_logit_init,
        "gate_nudge_weight=",
        args.gate_nudge_weight,
    )

    history = []
    step = 0
    model.train()
    t0 = time.time()
    while step < args.steps:
        for x, y in train_loader:
            if step >= args.steps:
                break
            x = x.to(device)
            y = y.to(device)
            pred, probe, gate = model(x)
            loss_main = torch.mean((pred - y) ** 2)
            loss_probe = torch.mean((probe - x) ** 2)
            loss = loss_main + args.probe_weight * loss_probe

            nudge_w = args.gate_nudge_weight if step < args.gate_nudge_steps else 0.0
            loss_gate = None
            if nudge_w > 0:
                loss_gate = (gate - args.gate_target) ** 2
                loss = loss + nudge_w * loss_gate

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            if step % 20 == 0 or step == args.steps - 1:
                msg = {
                    "step": step,
                    "loss": float(loss.item()),
                    "loss_main": float(loss_main.item()),
                    "loss_probe": float(loss_probe.item()),
                    "gate": float(gate.item()),
                }
                if loss_gate is not None:
                    msg["loss_gate"] = float(loss_gate.item())
                    msg["gate_nudge_w"] = float(nudge_w)
                history.append(msg)
                print(msg, flush=True)
            step += 1

    val_main, val_probe = evaluate(model, val_loader, device, args.probe_weight)
    final_gate = float(torch.sigmoid(model.gate_logit).item())
    elapsed = time.time() - t0
    print(
        f"val_mse_next={val_main:.6f} val_mse_probe={val_probe:.6f} "
        f"final_gate={final_gate:.4f} elapsed_s={elapsed:.1f}",
        flush=True,
    )

    ckpt = {
        "scalar_enc": model.scalar_enc.state_dict(),
        "gate_logit": model.gate_logit.detach().cpu(),
        "head": model.head.state_dict(),
        "probe": model.probe.state_dict(),
        "config": {
            "model_path": args.model,
            "hidden_size": hidden,
            "seq_len": args.seq_len,
            "steps": args.steps,
            "freeze_backbone": freeze,
            "probe_weight": args.probe_weight,
            "gate_logit_init": args.gate_logit_init,
            "gate_nudge_weight": args.gate_nudge_weight,
            "gate_target": args.gate_target,
            "gate_nudge_steps": args.gate_nudge_steps,
            "lr": args.lr,
            "batch_size": args.batch_size,
        },
        "history": history,
        "val_mse_next": val_main,
        "val_mse_probe": val_probe,
        "final_gate": final_gate,
    }
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(ckpt, ckpt_path)
    metrics = {
        "history": history,
        "val_mse_next": val_main,
        "val_mse_probe": val_probe,
        "final_gate": final_gate,
        "elapsed_s": elapsed,
        "steps": args.steps,
        "device": str(device),
        "cuda": torch.cuda.is_available(),
        "gate_logit_init": args.gate_logit_init,
        "gate_nudge_weight": args.gate_nudge_weight,
        "gate_target": args.gate_target,
        "gate_nudge_steps": args.gate_nudge_steps,
        "notes": (
            "Feasibility: higher gate init (default 0.0->0.5) vs smoke -2.0->0.12; "
            "optional early gate nudge; backbone frozen."
        ),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print("saved", ckpt_path, flush=True)


if __name__ == "__main__":
    main()
