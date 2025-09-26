import torch
from torch import nn

class ff(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(dropout, mult * d_model),
            nn.GELU(),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.f(x)
    