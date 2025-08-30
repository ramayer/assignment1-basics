
from .ron_softmax import softmax
import einx
from jaxtyping import Float, Int
import torch
from torch import Tensor

def cross_entropy(
        predicted_logits: Float[Tensor, " batch_size vocab_size"], 
        targets: Int[Tensor, " batch_size"]
    )-> Float[Tensor, ""]:
    """
            Deliverable: 
            Write a function to compute the cross entropy loss, which takes in 
            predicted logits (oi ) and 
            targets (xi+1 ) 
            and computes the cross entropy 
                ℓi = − log softmax(oi )[xi+1 ]. 
                
            Your function should handle the following:
            • Subtract the largest element for numerical stability.
            • Cancel out log and exp whenever possible.
            • Handle any additional batch dimensions and return the average across the batch. As with sec-
            tion 3.3, we assume batch-like dimensions always come first, before the vocabulary size dimension.
            Implement [adapters.run_cross_entropy], then run uv run pytest -k test_cross_entropy
            to test your implementation.


            inputs = torch.tensor(
                [
                    [
                        [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
                        [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
                        [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
                        [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
                    ],
                    [
                        [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
                        [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
                        [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
                        [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
                    ],
                ]
            )
            targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])
            expected = F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))
    
            Einx hint -- these are quite simlar:
                einx.get_at("... [c], ... -> ...", lsm, targets)
                einx.get_at("... [c], ... -> ... 1", lsm, targets)
                torch.gather(lsm, -1, targets.unsqueeze(-1))
                lsm[torch.arange(targets.shape[0]), targets]
    """

    #Subtract the largest element for numerical stability.
    maxes,_idxs = einx.max("... [d]",predicted_logits) # Why does it choke with ,keepdims=True ?
    maxes = einx.rearrange("... -> ... 1", maxes) # put the dim back
    assert isinstance(maxes,Tensor) # avoid vscode warning
    subtracted = predicted_logits - maxes
    print("sub = ",subtracted)
    ### 
    #x = subtracted
    #y = x - torch.log(torch.sum(torch.exp(x), dim=-1, keepdim=True))
    #z = torch.mean(torch.gather(y, -1, targets.unsqueeze(-1)))
    #return -z

    # This part fails on the 
    # "cross-entropy handles numerical overflow issues" test :(
    # E       +inf location mismatch:
    # E        ACTUAL: array(inf)
    # E        DESIRED: array(272.9625, dtype=float32)
    subtracted = subtracted.to(torch.float64)
    sm = softmax(subtracted,-1)
    lsm = torch.log(sm)

    # TODO - implement it right later, but workaround with pytorch
    import torch.functional as F
    lsm = torch.log_softmax(subtracted,-1)
    print("lsm = ",lsm)

    picked = einx.get_at("... [c], ... -> ... 1", lsm, targets)
    #print("good g ",g)
    # g = tensor([-5.6230e+02, -2.3140e+02,  0.0000e+00, -8.0540e+02, -6.7900e+01,
    #    -1.8990e+02, -3.2680e+02, -1.3323e-15], dtype=torch.float64
    z = torch.mean(picked)
    return z * -1
    


    print(subtracted)
    sm = softmax(subtracted,-1)
    print("SM ",sm)

    lsm = torch.log(sm)
    print("LSM ",lsm)

    total = -einx.mean("[...]",lsm)

    #print("MP ",max_predicted_logit)
    print("YEY?")
    #x = predicted_logits - max_predicted_logit

    #sm = softmax(predicted_logits,targets)
    return total