import torch
from __future__ import annotations


def top_k_top_p(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None):
    bs, vs = logits.shape
    fil = logits.clone()

    if top_k is not None and top_k < vs:
        topk_vals, _ = torch.topk(fil, top_k, dim=-1)
        kth = topk_vals[:, -1].unsqueeze(-1)
        fil[fil < kth] = float('-inf')

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(fil, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        mask = cumsum > top_p
        mask[..., 0] = False # mask[..., 0] is similar to mask[:, :, 0]
        sorted_logits[mask] = float('-inf')
        fil = torch.full_like(fil, float('-inf'))
        fil.scatter_(1, sorted_idx, sorted_logits)
    return fil