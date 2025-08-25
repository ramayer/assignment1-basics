import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
import einx
from .ron_rope import RoPE
from .ron_linear import Linear
from .ron_scaled_dot_product_attention import scaled_dot_product_attention

# test with:
#   pytest -k "test_multihead_self_attention and not with_rope"

class MultiheadSelfAttention(torch.nn.Module):
    """
        Deliverable: Implement causal multi-head self-attention as a torch.nn.Module. Your implemen-
        tation should accept (at least) the following parameters:
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        Folllowing Vaswani et al. [2017], set dk = dv = dmodel /h. To test your implementation against our
        provided tests, implement the test adapter at [adapters.run_multihead_self_attention]. Then,
        run uv run pytest -k test_multihead_self_attention to test your implementation.
    
    
        Here, the learnable parameters are WQ ∈ Rhdk ×dmodel , WK ∈ Rhdk ×dmodel , WV ∈ Rhdv ×dmodel , and WO ∈
        Rdmodel ×hdv . Since the Qs, K, and V s are sliced in the multi-head attention operation, we can think of WQ ,
        WK and WV as being separated for each head along the output dimension. When you have this working,
        you should be computing the key, value, and query projections in a total of three matrix multiplies.5
    
    """
    
    def __init__(self,
                 d_model: int, 
                 num_heads: int
        ):
        super().__init__()
        dk = dv = d_model // num_heads
        self.dk = dk
        self.dv = dv
        self.dm = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, num_heads * dk)
        self.k_proj = Linear(d_model, num_heads * dk)
        self.v_proj = Linear(d_model, num_heads * dv)
        self.o_proj = Linear(num_heads * dv, d_model)

    def make_a_triangle_mask(self,sequence_length:int):
        idx = torch.arange(sequence_length)
        triangle_mask = idx[:, None] >= idx[None, :]     # [L, L], boolean
        return triangle_mask

    def forward(self,x:Tensor):
        """
                in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = einx.rearrange("... seq_len (heads d) -> ... heads seq_len d",q,heads=self.num_heads)
        k = einx.rearrange("... seq_len (heads d) -> ... heads seq_len d",k,heads=self.num_heads)
        v = einx.rearrange("... seq_len (heads d) -> ... heads seq_len d",v,heads=self.num_heads)

        triangle_mask = self.make_a_triangle_mask(sequence_length=x.shape[-2])
        attn_output = scaled_dot_product_attention(k=k, q=q, v=v, mask=triangle_mask)

        attn_output = einx.rearrange("... heads seq d_v -> ... seq (heads d_v)",attn_output)
        output = self.o_proj(attn_output)
        return output
