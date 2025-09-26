import torch
from torch import nn
import torch.nn.functional as F
import math
from attn_mask import casual_mask


class single_head_atnn(nn.Module):
    def __init__(self, d_model: int, d_k: int, dropout: float = 0.0, trace_shape: bool = False):
        super().__init__()
        self.wq = nn.Linear(d_model, d_k, bias=False)
        self.wk = nn.Linear(d_model, d_k, bias=False)
        self.wv = nn.Linear(d_model, d_k, bias=False)
        self.dr = nn.Dropout(dropout)
        self.tr = trace_shape

    def forward(self, x: torch.Tensor):
        bs, sl, dim = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        if self.tr:
            print(f"q: {q.shape}")
        sc = 1.0 / math.sqrt(q.size(-1))
        attn = torch.matmul(q, k.transpose(-2, -1)) * sc
        mask = casual_mask(sl, device=x.device)
        attn = attn.masked_fill_(mask.squeeze(1), float('-inf'))
        w = F.softmax(attn, dim=-1)
        w = self.dr(w)
        out = torch.matmul(w, v)
        if self.tr:
            print(f"wei: {w.shape}")
        return out, w   


