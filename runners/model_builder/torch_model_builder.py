
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from runners.model_builder.model_builder import ModelBuilder
from runners.model_builder.models.torch.CNNComplex import CNNComplex
from runners.model_builder.models.torch.CNNSimple import CNNSimple
from runners.model_builder.models.torch.LSTMComplex import LSTMComplex
from runners.model_builder.models.torch.LSTMSimple import LSTMSimple
from runners.model_builder.models.torch.MLPComplex import MLPComplex
from runners.model_builder.models.torch.MLPSimple import MLPSimple
from utils.torch_utils import accuracy, mae


class TorchModelBuilder(ModelBuilder):

    def _mlp_simple(self):
        activation = nn.ReLU()
        dropout = 0.2
        lr = 1e-4
        
        model = MLPSimple(activation, dropout)

        config = {
            "optimizer": optim.Adam(model.parameters(), lr=lr),
            "loss_fn": nn.CrossEntropyLoss(),
            "metric_fn": accuracy,
            "metric_name": "accuracy"
        }

        return model, config
    

    def _mlp_complex(self):
        activation = nn.ReLU()
        dropout = 0.4
        lr = 1e-4

        model = MLPComplex(activation, dropout)
        
        config = {
            "optimizer": optim.Adam(model.parameters(), lr=lr),
            "loss_fn": nn.CrossEntropyLoss(),
            "metric_fn": accuracy,
            "metric_name": "accuracy"
        }

        return model, config


    def _cnn_simple(self):
        
        activation = nn.ReLU()
        dropout = 0.2
        lr = 1e-4

        model = CNNSimple(activation, dropout)
        
        config = {
            "optimizer": optim.Adam(model.parameters(), lr=lr),
            "loss_fn": nn.CrossEntropyLoss(),
            "metric_fn": accuracy,
            "metric_name": "accuracy"
        }

        return model, config
        


    def _cnn_complex(self):
        n = 10 # number of blocks, 6n + 1 layers
        lr = 1e-4

        model = CNNComplex(n_blocks=n)
        
        config = {
            "optimizer": optim.Adam(model.parameters(), lr=lr),
            "loss_fn": nn.CrossEntropyLoss(),
            "metric_fn": accuracy,
            "metric_name": "accuracy"
        }

        return model, config
    

    def _lstm_simple(self):        
        cells = 32
        dropout = 0.1
        lr = 1e-4

        model = LSTMSimple(cells, dropout)

        config = {
            "optimizer": optim.Adam(model.parameters(), lr=lr),
            "loss_fn": nn.MSELoss(),
            "metric_fn": mae,
            "metric_name": "mae"
        }

        return model, config

    def _lstm_complex(self):
        lstm_layers = 8
        initial_cells = 512
        dropout = 0.4
        lr = 1e-4

        model = LSTMComplex(lstm_layers, initial_cells, dropout)

        config = {
            "optimizer": optim.Adam(model.parameters(), lr=lr),
            "loss_fn": nn.MSELoss(),
            "metric_fn": mae,
            "metric_name": "mae"
        }

        return model, config