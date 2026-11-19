# Position causality check (CPU, no training):
# Does bar position q=2t+1 see only keys k<=2t+1?  And can it reach m_{t-L} at k=2(t-L)?
# Method: monkeypatch F.scaled_dot_product_attention on the full-window causal path,
#         recompute weights explicitly; plus a gradient check.
import sys, os
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

ROOT = Path(r"E:\rwkv7"); sys.path.insert(0, str(ROOT / "kronos_src"))
import model.module as mm
from model import Kronos

dev = "cpu"
torch.set_num_threads(int(os.environ.get("M4_THREADS", "4")))
T = int(os.environ.get("M4_T", "32")); L = int(os.environ.get("M4_L", "5"))
B = 1; LAYER = int(os.environ.get("M4_PL", "4"))

model = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).to(dev).eval()
for p in model.parameters(): p.requires_grad = False
H = model.d_model

REC = []
_orig = F.scaled_dot_product_attention
def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
    if is_causal:
        assert attn_mask is None, "causal path must not carry a mask"
        Lq, Lk = q.shape[-2], k.shape[-2]
        s = (q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        allow = torch.ones(Lq, Lk, dtype=torch.bool, device=q.device).tril()
        REC.append(s.masked_fill(~allow, float("-inf")).softmax(-1).detach())
    return _orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kw)
F.scaled_dot_product_attention = sdpa

z = np.load(ROOT / "runs" / "e6_long_2026-10-20" / "e5_tokens.npz")
S1 = torch.tensor(z["S1"][:T][None], device=dev); S2 = torch.tensor(z["S2"][:T][None], device=dev)
st = torch.zeros(B, T, 5, device=dev)
torch.manual_seed(0)
m = torch.stack([torch.randint(0, 2, (B, T)).float() * 2 - 1, torch.randn(B, T)], -1).to(dev)
enc_m = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.GELU(), torch.nn.Linear(64, H)).to(dev)

with torch.no_grad():
    xb = model.embedding([S1, S2]) + model.time_emb(st)
    xm = enc_m(m)
    x = torch.stack([xm, xb], dim=2).reshape(B, 2 * T, H)
print("interleave x[:,2t]==m:", torch.allclose(x[:, 0::2], xm), "| x[:,2t+1]==bar:", torch.allclose(x[:, 1::2], xb))

REC.clear()
x2 = x.clone().requires_grad_(True)
out = x2
for layer in model.transformer:
    out = layer(out)
out = model.norm(out)
p = REC[LAYER][0].mean(0)            # layer LAYER, head-mean, [2T,2T]

for t in [L, L + 1, T // 2, T - 1]:
    q = 2 * t + 1; k = 2 * (t - L); fut = p[q, q + 1:]
    print("t=%2d q=%2d k_target=%2d dist=%2d w_target=%.6f w_cur_m=%.6f w_cur_bar=%.6f max_future=%.8f row_sum=%.6f"
          % (t, q, k, q - k, p[q, k].item(), p[q, 2 * t].item(), p[q, 2 * t + 1].item(),
             fut.max().item() if fut.numel() else 0.0, p[q].sum().item()))

q0 = 2 * L + 1
print("min weight over allowed keys (q=%d): %.3e" % (q0, p[q0, :q0 + 1].min().item()))
print("max weight over disallowed keys (q=%d): %.3e"
      % (q0, p[q0, q0 + 1:].max().item() if q0 + 1 <= 2 * T else 0.0))

for tg in [L, L + 2, min(20, T - 2)]:
    g2 = torch.autograd.grad(out[0, 2 * tg + 1].sum(), x2, retain_graph=True)[0][0].abs().sum(-1)
    ktg = 2 * (tg - L); nfut = g2[2 * tg + 2:]
    print("q=%2d grad@k=2(t-L)=%2d %.3e | grad@k=2t(cur m) %.3e | future grads all zero: %s | max future grad %.3e"
          % (2 * tg + 1, ktg, g2[ktg].item(), g2[2 * tg].item(),
             bool((nfut.sum() == 0).item()) if nfut.numel() else True,
             nfut.max().item() if nfut.numel() else 0.0))
F.scaled_dot_product_attention = _orig