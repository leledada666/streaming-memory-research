# -*- coding: utf-8 -*-
import sys, time, torch
from pathlib import Path
ROOT = Path(r"E:\rwkv7"); sys.path.insert(0, str(ROOT / "kronos_src"))
from model import Kronos
torch.set_num_threads(8)
m = Kronos.from_pretrained(str(ROOT / "models" / "Kronos-small")).eval()
for p in m.parameters(): p.requires_grad = False
print("H", m.d_model, "NL", len(m.transformer))
for T, B in [(16, 2), (32, 2), (32, 4)]:
    s1 = torch.zeros(B, T, dtype=torch.long); s2 = torch.zeros(B, T, dtype=torch.long)
    st = torch.zeros(B, T, 5)
    with torch.no_grad():
        t0 = time.time()
        x = m.embedding([s1, s2]) + m.time_emb(st)
        for l in m.transformer: x = l(x)
        h = m.norm(x)
        dt = time.time() - t0
    print(f"T={T} B={B} fwd={dt*1000:.0f}ms")