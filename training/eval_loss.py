from __future__ import annotations
import torch
from torch import nn
from gpt import gpt1
from data import byte_ds
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    ds = byte_ds(args.data, block_size=args.block_size)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', {
        'vocab_size': 256,
        'block_size': args.block_size,
        'n_layer': 4,
        'n_head': 4,
        'n_embd': 256,
        'dropout': 0.0,
    })
    model = gpt1(**cfg).to(device)
    model.load_state_dict(ckpt['model'])

    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(args.iters):
            xb, yb = ds.get_batch('val', args.batch_size, device=device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        print(f"val loss: {sum(losses)/len(losses):.4f}")


if __name__ == "__main__":
    main()



