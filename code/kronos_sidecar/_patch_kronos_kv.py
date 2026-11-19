# -*- coding: utf-8 -*-
"""Patch Kronos module.py for grow-only KV cache with RoPE position offset."""
from pathlib import Path
import re

p = Path(r"E:/rwkv7/kronos_src/model/module.py")
text = p.read_text(encoding="utf-8")

rope_new = '''
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _update_cos_sin_cache(self, x, seq_len):
        # Grow-only cache of cos/sin for absolute positions [0, seq_len).
        if self.seq_len_cached is None or seq_len > self.seq_len_cached or self.cos_cached is None or self.cos_cached.device != x.device:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached, self.sin_cached

    def forward(self, q, k, position_offset=0):
        """Apply RoPE. position_offset = absolute start index of this q/k chunk (for KV cache)."""
        seq_len = q.shape[-2]
        end = position_offset + seq_len
        cos, sin = self._update_cos_sin_cache(q, end)
        cos = cos[:, :, position_offset:end, :].to(dtype=q.dtype)
        sin = sin[:, :, position_offset:end, :].to(dtype=q.dtype)
        return (
            (q * cos) + (self._rotate_half(q) * sin),
            (k * cos) + (self._rotate_half(k) * sin),
        )

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


'''

attn_new = '''
class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout_p=0.0, resid_dropout_p=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(self, x, key_padding_mask=None, past_kv=None, use_cache=False):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        position_offset = 0 if past_kv is None else past_kv[0].shape[-2]
        q, k = self.rotary(q, k, position_offset=position_offset)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=-2)
            v = torch.cat([past_kv[1], v], dim=-2)

        kv_len = k.shape[-2]
        if key_padding_mask is not None:
            # True = pad/ignore in Kronos convention for key_padding_mask? Original expands as attn_mask into SDPA.
            # Keep original semantics: pass through as additive/bool mask broadcast to [B,H,q,k].
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.n_heads, seq_len, -1)
            if past_kv is not None and attn_mask.shape[-1] != kv_len:
                # padding mask only covers current keys; for cached path prefer no pad in smoke
                raise ValueError("key_padding_mask length must match full kv_len when using cache")
        else:
            attn_mask = None

        # Full window: causal. Incremental single-token: attend to all past+self (exact).
        if past_kv is None:
            is_causal = True
        else:
            is_causal = False
            if seq_len > 1:
                # Build causal mask for chunked decode: query at abs pos can see keys <= that pos.
                q_pos = torch.arange(position_offset, position_offset + seq_len, device=x.device)[:, None]
                k_pos = torch.arange(kv_len, device=x.device)[None, :]
                # SDPA bool mask: True means NOT allowed to attend (PyTorch)
                disallow = k_pos > q_pos  # [q, k]
                attn_mask = disallow[None, None, :, :].expand(batch_size, self.n_heads, -1, -1)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=is_causal
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.resid_dropout(self.out_proj(attn_output))
        if use_cache:
            return out, (k, v)
        return out


'''

block_new = '''
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim=1024, ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionWithRoPE(d_model, n_heads, attn_dropout_p, resid_dropout_p)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ff_dim, ffn_dropout_p)

    def forward(self, x, key_padding_mask=None, past_kv=None, use_cache=False):
        residual = x
        x = self.norm1(x)
        if use_cache:
            attn_out, present = self.self_attn(x, key_padding_mask=key_padding_mask, past_kv=past_kv, use_cache=True)
        else:
            attn_out = self.self_attn(x, key_padding_mask=key_padding_mask, past_kv=past_kv, use_cache=False)
            present = None
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        if use_cache:
            return x, present
        return x


'''

def replace_class(src, class_name, new_body):
    # Replace from "class Name" until next top-level "class " or EOF
    pat = re.compile(rf"(class {class_name}\b.*?)(?=\nclass |\Z)", re.S)
    m = pat.search(src)
    if not m:
        raise SystemExit(f"class {class_name} not found")
    return src[:m.start()] + new_body.lstrip("\n") + src[m.end():]

text2 = replace_class(text, "RotaryPositionalEmbedding", rope_new)
text2 = replace_class(text2, "MultiHeadAttentionWithRoPE", attn_new)
text2 = replace_class(text2, "TransformerBlock", block_new)
p.write_text(text2, encoding="utf-8")
print("module.py patched")
for name in ["RotaryPositionalEmbedding", "MultiHeadAttentionWithRoPE", "TransformerBlock"]:
    assert f"class {name}" in text2
print("ok")
