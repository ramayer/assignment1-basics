import math
import torch

def gradient_clipping(params, max_l2_norm, eps = 1e-6):
    """
    Your function should take a list of parameters nd a maximum ℓ2 -norm. It should 
    modify each parameter gradient in place. Use ϵ = 10−6 (the PyTorch default).
    """
    ssq = sum([torch.sum(p.grad**2) for p in params if p.grad is not None])
    norm = math.sqrt(ssq)
    scale = min(1, max_l2_norm / (norm + eps))
    for p in params:
        if p.grad is not None:
            p.grad *= scale
