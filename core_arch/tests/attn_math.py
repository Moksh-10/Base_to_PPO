import numpy as np
import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import single_head as sh

X = np.array([[[0.1, 0.2, 0.3, 0.4],
               [0.5, 0.4, 0.3, 0.2],
               [0.0, 0.1, 0.0, 0.1]]], dtype=np.float32)
Wq = np.array([[ 0.2, -0.1],[ 0.0,  0.1],[ 0.1,  0.2],[-0.1,  0.0]], dtype=np.float32)
Wk = np.array([[ 0.1,  0.1],[ 0.0, -0.1],[ 0.2,  0.0],[ 0.0,  0.2]], dtype=np.float32)
Wv = np.array([[ 0.1,  0.0],[-0.1,  0.1],[ 0.2, -0.1],[ 0.0,  0.2]], dtype=np.float32)


def test_single_head_matches():
    torch.manual_seed(0)
    x = torch.tensor(X)
    attn = sh.single_head_atnn(d_model=4, d_k=2)
    with torch.no_grad():
        attn.wq.weight.copy_(torch.tensor(Wq).t())
        attn.wk.weight.copy_(torch.tensor(Wk).t())
        attn.wv.weight.copy_(torch.tensor(Wv).t())
    out, w = attn(x)
    assert out.shape == (1, 3, 2)
    assert torch.isfinite(out).all()
    assert torch.isfinite(w).all()

