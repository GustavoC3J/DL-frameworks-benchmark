
import torch


def accuracy(preds, y):
    _, predicted = torch.max(preds, 1) # Find highest. 1 is the dimension
    return (predicted == y).sum().item() / y.size(0)


def mae(preds, y):
    return torch.mean(torch.abs(preds - y)).item()