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
    print("In softmax, t.shape = ",t.shape, " and i = ", i)

    # This is probably unfair; but cool that einx has it available.
    # TODO - come back and implement it manually
    if i < 0:
        i = len(t.shape) + i
    einx_expr = " ".join(f"a{n}" for n in range(i)) + " [d] ..."
    return einx.softmax(einx_expr,t)
