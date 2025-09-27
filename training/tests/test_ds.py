import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import data


def test_shift_align(tmp_path):
    p = tmp_path / 'toy.txt'
    p.write_text('abcdefg')
    ds = byte_ds(str(p), block_size=3, split=1.0)
    x, y = ds.get_batch('train', 2, device=torch.device('cpu'))
    assert (y[:, :-1] == x[:, 1:]).all()
