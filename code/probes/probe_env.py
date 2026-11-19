import numpy as np, json, sys, torch
sys.path.insert(0, r"E:\rwkv7\kronos_src")
z = np.load(r"E:\rwkv7\runs\e6_long_2026-10-20\e5_tokens.npz")
print({k: (z[k].shape, str(z[k].dtype)) for k in z.files})
print(open(r"E:\rwkv7\models\Kronos-small\config.json").read())
from model import Kronos
m = Kronos.from_pretrained(r"E:\rwkv7\models\Kronos-small").eval()
print("d_model", m.d_model, "layers", len(m.transformer), "params", sum(p.numel() for p in m.parameters()))
print("head", m.transformer[0].self_attn.n_heads, m.transformer[0].self_attn.head_dim)