import torch
from multi_head import mha
from visuals import save_atnn_heads_grid

bs, sl, d_model, n_head = 1, 5, 12, 3
x = torch.randn(bs, sl, d_model)
attn = mha(d_model, n_head, trace_shape=False)
out, w = attn(x)
save_atnn_heads_grid(w.detach().cpu().numpy(), filename="multi_head_attn_grid.png")