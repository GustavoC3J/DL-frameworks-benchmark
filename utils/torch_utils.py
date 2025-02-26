
import torch
import numpy as np
from pandas import Series, DataFrame

def accuracy(preds, y):
    _, predicted = torch.max(preds, 1)
    return (predicted == y).sum().item() / y.size(0)

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