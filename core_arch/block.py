import torch
from torch import nn
from ffn import ff
from multi_head import mha

class tr_block(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = mha(d_model, n_head, dropout)
        self.f = ff(d_model, mult=4, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))[0]
        x = x + self.f(self.ln2(x))
        return x