
import torch


def accuracy(preds, y):
    _, predicted = torch.max(preds, 1) # Find highest. 1 is the dimension
    return (predicted == y).sum().item() / y.size(0)


def mae(preds, y):
    return torch.mean(torch.abs(preds - y)).item()


def adjust_outputs(outputs: torch.Tensor, targets: torch.Tensor):
    if outputs.shape == targets.shape:
        return outputs
    else:
        return outputs.view_as(targets)
