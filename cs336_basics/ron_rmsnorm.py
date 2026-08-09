import einx
import torch
from torch import nn

"""
Problem (rmsnorm): Root Mean Square Layer Normalization(1 point)

Deliverable: Implement RMSNorm as a torch.nn.Module.

We recommend the following interface:

def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)

Construct the RMSNorm module.

This function should accept the following parameters:
    d_model: int Hidden dimension of the model
    eps: float = 1e-5 Epsilon value for numerical stability
    device: torch.device | None = None Device to store the parameters on
    dtype: torch.dtype | None = None Data type of the parameters

def forward(self, x: torch.Tensor) -> torch.Tensor

Process an input tensor of shape (batch_size, sequence_length, d_model)
and return a tensor of the same shape.

Note: Remember to upcast your input to torch.float32 before performing the normalization (and
later downcast to the original dtype), as described above.

To test your implementation, implement the test adapter at [adapters.run_rmsnorm]. Then, run uv
run pytest -k test_rmsnorm.

        ...
        Here, g is a learnable “gain” parameter (there are d_model such parameters total),
        and ε is a hyperparameter that is often fixed at 1e-5.
"""


"""Hints

    From the einx docs, layernorm looks like this

    mean = einx.mean("... [c]", x, keepdims=True)
    var = einx.var("... [c]", x, keepdims=True)
    x = (x - mean) * torch.rsqrt(var + epsilon)

    x = einx.add("... [c]", x, einn.param(name="bias"))
    x = einx.multiply("... [c]", x, einn.param(name="scale"))

"""


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        g = torch.ones(d_model, device=device, dtype=dtype)
        self.g = nn.Parameter(g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x = tensor of shape (batch_size, sequence_length, d_model)

        Notes:

        Einx hint:
          einx.dot("... , ... -> ...", x, x)
        is the same as
          x ** 2

        Einx hint:
            einx.mean("... [d_model]",x2,keepdims=True)
        is the same as
            x2.mean(-1, keepdim=True)
        and this might be easier to read
            einx.mean("batch_size sequence_length [d_model]", x2, keepdims=True)
        """

        result_type = x.dtype
        x32 = x.to(torch.float32)
        x2 = einx.dot("... , ... -> ...", x32, x32)
        xm = einx.mean("... [d_model]", x2, keepdims=True)
        xr = torch.rsqrt(xm + self.eps)
        xo = x * xr
        result = self.g * xo
        return result.to(result_type)
