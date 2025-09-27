from __future__ import annotations
import torch
from torch import nn
from pathlib import Path


class byte_ds:
    def __init__(self, path: str, block_size: int = 256, split: float = 0.9):
        data = Path(path).read_bytes()
        data = torch.tensor(list(data), dtype=torch.long)
        n = int(len(data) * split)
        self.tr = data[:n]
        self.val = data[n:]
        self.bs = block_size

    def get_batch(self, which: str, batch_size: int, device: torch.device):
        buf = self.tr if which == 'train' else self.val
        assert len(buf) > self.bs + 1, 'file is small'
        idx = torch.randint(0, len(buf) - self.bs - 1, (batch_size,))
        x = torch.stack([buf[i:i+self.bs] for i in idx])
        y = torch.stack([buf[i+1:i+1+self.bs] for i in idx])
        return x.to(device), y.to(device)




