import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
import math
import einx

class RoPE(torch.nn.Module):
    
    def __init__(self,
                 theta: float, # some llms hard code this to 10000(?!?)
                 d_k: int, # head dimension
                 max_seq_len: int, 
                 device=None
                 ):
        """
        From the class PDF:
            angle(i,k) = i / (theta ^ 2k/d)
        Hint from a chatbot: 
           So in __init__, you’ll precompute sin/cos 
           tables with shape: [max_seq_len, d_k]

        Einx hint:
            einx.rearrange("a -> a b",t,b=1)
        seems the same as
            t[:,None]

        Einx hint:
            einx.rearrange("a -> b a",t,b=1)
        seems the same as
            t[None,:]


        """
        super().__init__()

        k = torch.arange(d_k//2, device=device)           # dimensions; shape (d_k//2,)
        i = torch.arange(max_seq_len, device=device)   # positions; shape (max_seq_len,)

        # reshape to make rectangles
        k = einx.rearrange("a -> one a",k,one=1) # shape(1,d_k//2)
        i = einx.rearrange("a -> a one",i,one=1) # shape(max_seq_len,1)

        assert isinstance(k, torch.Tensor) # just to make VS Code not complain
        assert isinstance(i, torch.Tensor) # just to make VS Code not complain

        exponents = 2 * k / d_k
        denominators = theta ** exponents

        angles = i / denominators       # (max_seq_len, d_k//2)

        s = torch.sin(angles)
        c = torch.cos(angles)


        self.sin_tbl: torch.Tensor # make VSCode not complain
        self.cos_tbl: torch.Tensor # make VSCode not complain

        print("s,c ",s.shape,c.shape)
        self.register_buffer("sin_tbl", s, persistent=False)
        self.register_buffer("cos_tbl", c, persistent=False)
        pass

    def forward(self, x: torch.Tensor, # shape (batch,seq_len,d_k), (4,12,64)
                token_positions: torch.Tensor # shape(12,)
                ) -> torch.Tensor:

        print("in forward: "
              "x ", x.shape, ", " \
              "token_positions", token_positions.shape)

        even_x = x[..., 0::2]  # dims 0,2,4,...
        odd_x  = x[..., 1::2]  # dims 1,3,5,...

        s = self.sin_tbl[token_positions]  # shape [batch, seq_len, d_k//2]
        c = self.cos_tbl[token_positions]  # shape [batch, seq_len, d_k//2]

        print("in forward: "
              "s ", s.shape, ", " \
              "c ", c.shape, ", "
              )

        new_even_x = even_x * c - odd_x * s
        new_odd_x  = odd_x  * c + even_x * s

        print("in forward: "
              "new_even_x ", new_even_x.shape, ", " 
              "new_odd_x ", new_odd_x.shape, ", "
              )
        x_paired = torch.stack([new_even_x, new_odd_x], dim=-1) # (batch,seq_len,d_k//2,2)
        # note, the multihead test requires "..." instead of "b" because the heads are also a batchlike dimension
        result = einx.rearrange("... seq d2 p -> ... seq (d2 p)",x_paired)  

        assert isinstance(result, torch.Tensor) # just to make VS Code not complain
        return result
        pass


