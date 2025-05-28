import os
import getpass
from datetime import datetime
import torch
import random
import numpy as np
import os


def get_local_dir(prefixes_to_resolve):
    for prefix in prefixes_to_resolve:
        if os.path.exists(prefix):
            return f"{prefix}/{getpass.getuser()}"
    os.makedirs(prefix)
    return f"{prefix}/{getpass.getuser()}"
    

def get_local_run_dir(exp_name, local_dirs):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir = f"{get_local_dir(local_dirs)}/{exp_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def slice_and_move_batch_for_device(batch, rank, world_size, device):
    chunk_size = len(list(batch.values())[0]) // world_size
    start = chunk_size * rank
    end = chunk_size * (rank + 1)
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()}
    return on_device


def pad_to_length(tensor, length, pad_value, dim = -1):
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)
    

def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


class TemporarilySeededRandom:
    def __init__(self, seed):
        self.seed = seed
        self.stored_state = None
        self.stored_np_state = None

    def __enter__(self):
        self.stored_state = random.getstate()
        self.stored_np_state = np.random.get_state()

        random.seed(self.seed)
        np.random.seed(self.seed)

    def __exit__(self):
        random.setstate(self.stored_state)
        np.random.set_state(self.stored_np_state)
