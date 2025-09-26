import math
import torch
from torch import nn
import torch.nn.functional as F
from attn_mask import casual_mask


class mha(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0, trace_shape: bool = True):
        super().__init__()
        assert d_model % n_head == 0, "must be divisible"
        self.n_head = n_head
        self.dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dr = nn.Dropout(dropout)
        self.tr = trace_shape

    def forward(self, x: torch.Tensor):
        bs,sl, dm = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bs, sl, 3, self.n_head, self.dim)
        if self.tr:
            print(f"qkv: {qkv.shape}")
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2)
        if self.tr:
            print(f"q, k, v: {q.shape}")
        sc = 1.0 / math.sqrt(self.dim)
        attn = torch.matmul(q, k.transpose(-1, -2)) * sc
        mask = casual_mask(sl, device=x.device)
        attn = attn.masked_fill_(mask, float('-inf'))
        w = F.softmax(attn, dim=-1)
        w = self.dr(w)
        o = torch.matmul(w, v)
        if self.tr:
            print(f"wei: {w.shape}")
        out = o.transpose(1, 2).contiguous().view(bs, sl, dm)
        out = self.proj(out)
        if self.tr:
            print(f"out: {out.shape}")
        return out, w   

