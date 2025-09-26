import torch
from torch import nn
import math


class learned_pos_encoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(seq_len, d_model)

    def forward(self, x):
        bs, sl, d_model = x.shape
        pos_emb = self.emb(torch.arange(sl, device=x.device))
        return x + pos_emb.unsqueeze(0)


class sin_pos_enc(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        bs, sl, dim = x.shape
        return x + self.pe[:sl].unsqueeze(0)


