import torch

def generate_causal_mask(size):
    mask = torch.ones(size, size)
    mask = torch.triu(mask, diagonal=1)
    return mask == 0