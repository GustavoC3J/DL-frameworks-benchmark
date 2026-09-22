
import torch
from torch import nn
from torch.nn import functional as F


# Keras computes the loss in float32 whatever the dtype policy; torch would use the compute dtype
def softmax_cross_entropy(logits, y):
    return F.cross_entropy(logits.float(), y)


def mse(preds, y):
    return F.mse_loss(preds.float(), y.float())


# Metrics return tensors to avoid that the CPU waits for the GPU
def accuracy(preds, y):
    _, predicted = torch.max(preds, 1) # Find highest. 1 is the dimension
    return (predicted == y).float().mean()


def mae(preds, y):
    return torch.mean(torch.abs(preds.float() - y.float()))


def init_lstm_weights(lstm):
    """Keras' LSTM initialization: glorot kernel, orthogonal recurrent kernel and unit_forget_bias.

    The QR of the orthogonal initializer runs here in float32, before the model is cast to the
    target precision, which float16 and bfloat16 do not support.
    """
    for name, parameter in lstm.named_parameters():
        if name.startswith("weight_ih"):
            nn.init.xavier_uniform_(parameter)
        elif name.startswith("weight_hh"):
            nn.init.orthogonal_(parameter)
        elif name.startswith("bias"):
            nn.init.zeros_(parameter)

    # Gate order (i, f, g, o): the forget gate of one of the two biases carries Keras' unit_forget_bias
    units = lstm.hidden_size
    with torch.no_grad():
        lstm.bias_ih_l0[units:2 * units].fill_(1.0)


def init_layer_weights(layer, kernel_initializer):

    # Initialize layer's weights using selected initializer
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        if kernel_initializer == "glorot_uniform":
            nn.init.xavier_uniform_(layer.weight)
        elif kernel_initializer == "glorot_normal":
            nn.init.xavier_normal_(layer.weight)
        elif kernel_initializer == "he_uniform":
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        elif kernel_initializer == "he_normal":
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        else:
            raise ValueError(f"Initializer not supported: {kernel_initializer}")
        
        # If bias is present, initialize to zeros
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

            
def adjust_outputs(outputs: torch.Tensor, targets: torch.Tensor):
    if outputs.shape == targets.shape:
        return outputs
    else:
        return outputs.view_as(targets)
