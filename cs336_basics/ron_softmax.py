import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
import math
import einx

def softmax(t:Tensor, i:int):
    """
    Problem (softmax): Implement softmax
    (1 point)
    Deliverable: Write a function to apply the softmax operation on a tensor. Your function should
    take two parameters: a tensor and a dimension i, and apply softmax to the i-th dimension of the input
    tensor. The output tensor should have the same shape as the input tensor, but its i-th dimension will
    now have a normalized probability distribution. Use the trick of subtracting the maximum value in
    the i-th dimension from all elements of the i-th dimension to avoid numerical stability issues.
    """

    maxes, _ = torch.max(t, dim=i, keepdim=True)
    e = torch.exp(t - maxes)
    return e / e.sum(dim=i, keepdim=True)

    """
    Note that Einx is more painful here where the question
    wants numbered dimensions rather than named ones.

    With einx it would look roughly like one fo these: 

        # This is probably unfair; but cool that einx has it available.
        # TODO - come back and implement it manually
        if i < 0:
            i = len(t.shape) + i
        einx_expr = " ".join(f"a{n}" for n in range(i)) + " [d] ..."
        return einx.softmax(einx_expr,t)

    
        # Subtract the largest element for numerical stability.
        maxes,_idxs = einx.max("... [vocab]",t) # Why does it choke with ,keepdims=True ?
        maxes = einx.rearrange("... -> ... 1", maxes) # put the dim back
        assert isinstance(maxes,Tensor) # avoid vscode warning
        subtracted = t - maxes
        
        # Compute the log softmax
        e = torch.exp(subtracted)
        s = einx.sum("... vocab -> ... 1", e) # einx.sum("... [vocab]", e, keepdims=True)
        sm = e / s
        return sm
    """




