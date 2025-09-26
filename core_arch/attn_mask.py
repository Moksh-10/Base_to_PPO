import torch

def casual_mask(sl: int, device=None):
    mask = torch.triu(torch.ones((sl, sl), dtype=torch.bool, device=device), diagonal=1)
    return mask.view(1, 1, sl, sl)

if __name__ == "__main__":
    b = casual_mask(3)
    print(b, b.shape)