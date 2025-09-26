import math
import torch
from torch import nn
from __future__ import annotations
import torch.nn.functional as f

class att(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.nh = n_head
        self.dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_head, bias=False)
        self.dr = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        bs, sl, dm = x.shape
        qkv = self.qkv(x).view(bs, sl, 3, self.nh, self.dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2)
        sc = 1.0 / math.sqrt(self.dim)
        y = f.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dr.p if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(bs, sl, dm)
        y = self.proj(y)
        return y


class ff(nn.Module):
    def __init__(self, n_embd: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(n_embd, mult * n_embd),
            nn.GELU(),
            nn.Linear(mult * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.f(x)


class block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = att(n_embd, n_head, dropout)
        self.ff = ff(n_embd, mult=4, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class gpt1(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layer: int = 4, n_head: int = 4, n_embd: int = 256, dropout: float = 0.0):
        super().__init__()
        self.bs = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block, n_embd)
        self.dr = nn.Dropout(dropout)
        self.bls = nn.ModuleList([block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        bs, sl = idx.shape
        assert sl <= self.bs
        pos = torch.arange(0, sl, device=idx.device).unsqueeze(0)
        x = self.tok_emb(x) + self.pos_emb(pos)
        x = self.dr(x)
        for b in self.bls:
            x = b(x)
        x = self.ln(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = f.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tok: int = 200, temp: float = 1.0, top_k: int | None = 50, top_p: float | None = None):
        from top_pk import top_k_top_p
        self.eval()
        if idx.size(1) == 0:
            idx = torch.full((idx.size(0), 1), 10, dtype=torch.long, device=idx.device)
        for _ in range(max_new_tok):
            idx_cond = idx[:, -self.bs:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temp, 1e-6)
            logits = top_k_top_p(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            new_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, new_id], dim=1)
        return idx
