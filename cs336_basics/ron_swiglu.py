import math

import torch
from torch import nn

from . import ron_linear


class SwiGLU(nn.Module):
    """
        Specifically, we will implement the “SwiGLU” activation function adopted 
        in LLMs like Llama 3 [Grattafiori et al., 2024] and Qwen 2.5 [Yang
        et al., 2024],

        ...
        SiLU(x) = x · σ(x)
        ...

        Gated Linear Units (GLUs) were originally defined by Dauphin et al. [2017] 
        as the element-wise product of a linear transformation passed through a 
        sigmoid function and another linear transformation:
        GLU(x, W1 , W2 ) = σ(W1 x) ⊙ W2 x,

        ...
        Putting the SiLU/Swish and GLU together, we get the SwiGLU, which we will 
        use for our feed-forward networks:
            FFN(x) = SwiGLU(x, W1 , W2 , W3 ) = W2 (SiLU(W1 x) ⊙ W3 x),
        where 
            x ∈ Rdmodel
            W1 , W3 ∈ Rdff × dmodel , 
            W2 ∈ Rdmodel ×dff , 
            and canonically, dff = 8/3 dmodel
        ...

        Deliverable: Implement the SwiGLU feed-forward network, composed of a SiLU activation
        function and a GLU.

        Note: in this particular case, you should feel free to use torch.sigmoid in your implementation
        for numerical stability.

        You should set dff to approximately 8/3 × dmodel in your implementation, while ensuring that
        the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your
        hardware. To test your implementation against our provided tests, you will need to implement
        the test adapter at [adapters.run_swiglu]. Then, run uv run pytest -k test_swiglu to
        test your implementation.
    """

    def __init__(self, d_model: int, dff: int | None):
        super().__init__()
        if dff is None:
            dff = math.ceil(d_model * 8 / 3 / 64) * 64
        self.w1 = ron_linear.Linear(d_model, dff)
        self.w2 = ron_linear.Linear(dff, d_model)
        self.w3 = ron_linear.Linear(d_model, dff)

    def silu(self,x):
        sx = torch.sigmoid(x)
        return x * sx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_w1_x = self.silu(self.w1(x)) # SiLU(W1 x)
        w3_x = self.w3(x) # (W3 x)
        gated = silu_w1_x * w3_x # (SiLU(W1 x) ⊙ W3 x)
        return self.w2(gated)
