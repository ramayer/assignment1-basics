import torch
from jaxtyping import Float, Int
from torch import Tensor, nn


class Embedding(torch.nn.Module):
    """
        ...
        As discussed above, the first layer of the Transformer is an embedding layer that maps integer token IDs
        into a vector space of dimension d_model. We will implement a custom Embedding class that inherits from
        torch.nn.Module (so you should not use nn.Embedding). The forward method should select the embedding
        vector for each token ID by indexing into an embedding matrix of shape (vocab_size, d_model) using a
        torch.LongTensor of token IDs with shape (batch_size, sequence_length).
        ...
        Deliverable: Implement the Embedding class that inherits from torch.nn.Module
        and performs an embedding lookup. Your implementation should follow the
        interface of PyTorch’s built-in nn.Embedding module. We recommend the
        following interface:
        def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)
        ...
        Make sure to:
        • subclass nn.Module
        • call the superclass constructor
        • initialize your embedding matrix as a nn.Parameter
        • store the embedding matrix with the d_model being the final dimension
        • of course, don’t use nn.Embedding or nn.functional.embedding
        ...
        Again, use the settings from above for initialization, and use torch.nn.init.trunc_normal_ to
        initialize the weights
        ...
        Some of their assignment's wording is comfusing...
        They inconsistently refer to d_model and embedding_dim in the assignment.
    """

    def __init__(self,
                 num_embeddings:int,
                 embedding_dim:int,
                 device:torch.device|None=None,
                 dtype:torch.dtype|None=None,
                 winput=None
                 ):
        super().__init__()
        if winput is not None:
            # wut's that undocumented "weights" parameter in the adapter?!?
            assert(winput.shape == torch.Size([num_embeddings, embedding_dim]))
            w = winput
        else:
            w = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
            w = nn.init.trunc_normal_(w, std=1.0, a=-3.0, b=3.0)
        self.weight = nn.Parameter(w)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids, :]
