import torch
import random


def random_update(keys):
    return {k: random.random() for k in keys}


def mse_values(keys):
    size = [4, 20]
    return {k: [torch.randn(*size), torch.randn(*size)] for k in keys}