import numpy.typing as npt
from jaxtyping import Float, Int
import random
import torch
import numpy as np
def get_batch(
        dataset: npt.NDArray, 
        batch_size: int, 
        context_length: int, 
        device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:

    """
    A data loader turns this into a stream of batches, where each batch consists of B sequences of length
    m, paired with the corresponding next tokens, also with length m. For example, for B = 1, m = 3,
    ([x2 , x3 , x4 ], [x3 , x4 , x5 ]) would be one potential batch.


    /home/ron/proj/cs336/assignment1-basics/cs336_basics/ron_data_loader.py:24: UserWarning: 
    Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider 
    converting the list to a single numpy.ndarray with numpy.array() before 
    converting to a tensor.

    """
    sequences = []
    next_seqs = []
    max_idx = dataset.size - context_length - 1
    for i in range(batch_size):
        idx = random.randint(0,max_idx)
        sequences.append(dataset[idx:idx+context_length])
        next_seqs.append(dataset[idx+1:idx+context_length+1])
    return (torch.tensor(np.array(sequences)).to(device), 
            torch.tensor(np.array(next_seqs)).to(device))