import math

import torch
from torch import nn
import torch.nn.functional as F


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)
    return out.to(x.dtype)


class CustomBlock(nn.Module):
    """
    Resolvent attention: replaces A @ V with (A + I)^-1 @ V, where A = softmax(QK^T).
    No separate MLP needed - the resolvent acts as the "mixing" operation.
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()
        cos, sin = cos_sin

        # Project to Q, K, V (no input norm - resolvent acts as normalization)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Rotary embeddings on Q, K only (no QK norm)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # (B, T, H, D) -> (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Handle GQA: expand k, v heads to match q heads
        if self.n_head != self.n_kv_head:
            n_rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # KV cache handling
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        Tq, Tk = q.size(2), k.size(2)

        # Compute attention logits: (B, H, Tq, Tk)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask (set future to -inf for softmax)
        if Tq == Tk:
            causal_mask = torch.triu(torch.full((Tq, Tk), float('-inf'), device=x.device), diagonal=1)
        elif Tq == 1:
            causal_mask = torch.zeros(1, Tk, device=x.device)
        else:
            causal_mask = torch.zeros(Tq, Tk, device=x.device)
            prefix_len = Tk - Tq
            causal_mask[:, prefix_len:] = torch.triu(torch.full((Tq, Tq), float('-inf'), device=x.device), diagonal=1)

        attn_logits = attn_logits + causal_mask

        # Softmax to get attention weights A
        A = F.softmax(attn_logits, dim=-1)  # (B, H, Tq, Tk)

        # Resolvent: (A + I)^-1 @ V instead of A @ V
        I = torch.eye(Tk, device=x.device, dtype=A.dtype)
        A_plus_I = A + I  # (B, H, Tq, Tk) + (Tk, Tk) - but this only works when Tq == Tk

        if Tq == Tk:
            # Square system: solve (A + I) @ y = v
            y = torch.linalg.solve(A_plus_I, v)  # (B, H, T, D)
        else:
            # Inference: build full matrix with identity for prefix rows
            full_A = torch.eye(Tk, device=x.device, dtype=A.dtype).unsqueeze(0).unsqueeze(0)
            full_A = full_A.expand(B, self.n_head, -1, -1).clone()
            full_A[:, :, -Tq:, :] = A_plus_I
            y_full = torch.linalg.solve(full_A, v)
            y = y_full[:, :, -Tq:, :]

        # Reassemble heads and project
        y = y.transpose(1, 2).contiguous().view(B, Tq, -1)
        y = self.c_proj(y)

        return x[:, -Tq:, :] + y  # Residual connection

