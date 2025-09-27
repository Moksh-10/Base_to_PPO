from __future__ import annotations
import torch
from torch import nn
import argparse
from gpt import gpt1
from tokenizer import byte_tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--prompt', type=str, default='')
    p.add_argument('--tokens', type=int, default=200)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--top_p', type=float, default=None)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    tok = byte_tok()
    prompt_ids = tok.encode(args.prompt).unsqueeze(0).to(device)
    if prompt_ids.numel() == 0:
        prompt_ids = torch.tensor([[10]], dtype=torch.long, device=device)

    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt.get('config', None)

    if config is None:
        model = gpt1(tok.vocab_size, block_size=256).to(device)
        model.load_state_dict(ckpt['model'])
    else:
        model = gpt1(**config).to(device)
        model.load_state_dict(ckpt['model'])

    with torch.no_grad():
        out = model.generate(prompt_ids, max_new_tok=args.tokens, temp=args.temperature, top_k=args.top_k, top_p=args.top_p)
    print(tok.decode(out[0].cpu()))


if __name__ == "__main__":
    main()

