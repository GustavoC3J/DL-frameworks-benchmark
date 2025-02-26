
import torch
import numpy as np
from pandas import Series, DataFrame

def accuracy(preds, y):
    _, predicted = torch.max(preds, 1) # Find highest. 1 is the dimension
    return (predicted == y).sum().item() / y.size(0)


def mae(preds, y):
    return torch.mean(torch.abs(preds - y)).item()


def to_float_tensor(array, device):
    if isinstance(array, (Series, DataFrame)):
        array = array.to_numpy()

    array = np.asarray(array, dtype=np.float32)

    return torch.from_numpy(array).to(device)


def to_long_tensor(array, device):
    if isinstance(array, (Series, DataFrame)):
        array = array.to_numpy()

    array = np.asarray(array, dtype=np.int64)

    return torch.from_numpy(array).to(device)