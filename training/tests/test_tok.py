import  torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tokenizer


def test():
    tok = tokenizer.byte_tok()
    s = "Hi there! äö"
    ids = tok.encode(s)
    assert ids.dtype == torch.long
    s2 = tok.decode(ids)
    assert len(s2) > 0


if __name__=="__main__":
    test()
