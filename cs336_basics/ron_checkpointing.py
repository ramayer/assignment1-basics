import torch
import torch.nn as nn
import torch.optim as optim
import typing
import os

def save_checkpoint(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    ):
    """
        should dump all the state from the first three parameters
        into the file-like object out. You can use the state_dict 
        method of both the model and the optimizer to get their 
        relevant states and use torch.save(obj, out) to dump obj 
        into out (PyTorch supports either a path or a file-like 
        object here). A typical choice is to have obj be a 
        dictionary, but you can use whatever format you want as 
        long as you can load your checkpoint later
    """
    #print("SC",model,optimizer,iteration,out)
    obj = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ):
    obj = torch.load(src)
    model.load_state_dict(obj["model_state_dict"])
    optimizer.load_state_dict(obj["optimizer_state_dict"])
    iteration = obj["iteration"]
    return iteration