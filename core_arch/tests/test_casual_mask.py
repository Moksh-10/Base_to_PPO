import numpy as np
import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import attn_mask

def test_is_upper():
    m = attn_mask.casual_mask(5)
    assert m.shape == (1, 1, 5, 5)
    assert m[0, 0].sum() == torch.triu(torch.ones(5, 5), diagonal=1).sum()

if __name__ == "__main__":
    test_is_upper()
    print("done")