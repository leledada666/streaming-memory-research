# -*- coding: utf-8 -*-
"""Add Kronos.forward_hidden with KV cache + optional bypass_delta."""
from pathlib import Path
import re

p = Path(r"E:/rwkv7/kronos_src/model/kronos.py")
text = p.read_text(encoding="utf-8")
Path(r"E:/rwkv7/kronos_src/model/kronos.py.bak").write_text(text, encoding="utf-8")

method = '''
    def forward_hidden(self, s1_ids, s2_ids, stamp=None, padding_mask=None,
                       past_key_values=None, use_cache=False, bypass_delta=None):
        """Embed tokens (+ optional time / bypass), run transformer, return hidden (+ cache).

        bypass_delta: optional [B, T, d_model] added after embedding (gated residual lives outside).
        past_key_values: tuple of per-layer (K, V), each K/V [B, H, S, Dh] post-RoPE K + raw V.
        Grow-only: do not slide the window while using cache (RoPE absolute positions).
        """
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.time_emb(stamp)
        if bypass_delta is not None:
            x = x + bypass_delta
        x = self.token_drop(x)

        presents = [] if use_cache else None
        for i, layer in enumerate(self.transformer):
            past = None if past_key_values is None else past_key_values[i]
            if use_cache:
                x, present = layer(x, key_padding_mask=padding_mask, past_kv=past, use_cache=True)
                presents.append(present)
            else:
                x = layer(x, key_padding_mask=padding_mask, past_kv=past, use_cache=False)
        x = self.norm(x)
        if use_cache:
            return x, tuple(presents)
        return x

'''

# Insert before decode_s1
anchor = "    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):"
if "def forward_hidden" in text:
    print("forward_hidden already present")
else:
    if anchor not in text:
        raise SystemExit("decode_s1 anchor not found")
    text = text.replace(anchor, method + anchor)
    p.write_text(text, encoding="utf-8")
    print("kronos.py patched")
