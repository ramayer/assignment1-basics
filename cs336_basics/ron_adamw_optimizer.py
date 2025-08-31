from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):

    """
        
        Deliverable: Implement the AdamW optimizer as a subclass of torch.optim.Optimizer. Your
        class should take the learning rate α in __init__, as well as the β, ϵ and λ hyperparameters. To help
        you keep state, the base Optimizer class gives you a dictionary self.state, which maps nn.Parameter
        objects to a dictionary that stores any information you need for that parameter (for AdamW, this would
        be the moment estimates). Implement [adapters.get_adamw_cls] and make sure it passes uv run
        pytest -k test_adamw.

        def __init__(self, 
                    α = 0.001, 
                    β = (0.9, 0.999), 
                    ϵ = 1e-8,
                    λ = 0.01
                    )
    """
    
    def __init__(self,
                 params,
                 lr = 1e-3, 
                 betas = (0.9, 0.999), 
                 weight_decay = 0.01,
                 eps = 1e-8,
                 ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> float|None: # type: ignore
        #print("in step")
        loss = None if closure is None else closure()

        for group in self.param_groups:
            # same as the class's SGD example
            lr    = group["lr"] # Get the learning rate.
            b1,b2 = group["betas"]
            wd    = group["weight_decay"]
            eps   = group["eps"]
            
            for p in group["params"]:
                # the first parts are identical to the SGD exmaple
                if p.grad is None:
                    continue

                # g ← ∇θ ℓ(θ; Bt ) (Compute the gradient of the loss at the current time step)
                grad = p.grad.data # Get the gradient of loss with respect to p.
                state = self.state[p] # Get state associated with p.
                #print("state ",state)

                # for t = 1, . . . , T do
                t = state.get("t", 1) # Get iteration number from the state, or initial value.

                ## From the assignment
                # m ← 0 (Initial value of the first moment vector; same shape as θ)
                # v ← 0 (Initial value of the second moment vector; same shape as θ)
                m = state.get("m",torch.zeros_like(p.data))
                v = state.get("v",torch.zeros_like(p.data))

                # m ← β1 m + (1 − β1 )g (Update the first moment estimate)
                new_m = b1 * m + grad*(1 - b1)
                # v ← β2 v√+ (1 − β2 )g^2 (Update the second moment estimate)
                new_v = b2 * v + grad*grad*(1 - b2)

                # αt ← α sqrt(1-(β1)^t) / (1-β1^t)
                at = lr * math.sqrt(1-b2**t) / (1-b1**t)

                # θ ← θ − αt * m / (√(v)+ϵ)
                p.data -= at * new_m / (torch.sqrt(new_v) + eps)

                # θ ← θ − αλθ (Apply weight decay )
                p.data -= lr * wd * p.data
                
                state['m'] = new_m
                state['v'] = new_v
                state["t"] = t + 1 # Increment iteration number.
        return loss
    
