
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from runners.model_builder.model_builder import ModelBuilder
from runners.model_builder.models.torch.cnn_complex import CNNComplex
from runners.model_builder.models.torch.cnn_simple import CNNSimple
from runners.model_builder.models.torch.lstm_complex import LSTMComplex
from runners.model_builder.models.torch.lstm_simple import LSTMSimple
from runners.model_builder.models.torch.mlp_complex import MLPComplex
from runners.model_builder.models.torch.mlp_simple import MLPSimple
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
        lr = 3e-5

        model = MLPComplex(
            groups=5,
            layers_per_group=2,
            dropout=0.2,
            initial_units=800,
            final_units=160,
            kernel_initializer="he_uniform"
        )
        
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
        lr = 1e-3

        model = CNNComplex(
            n_blocks=5, # number of blocks, 6n + 2 layers
            starting_channels=64,
            kernel_initializer="he_uniform"
        )
        
        config = {
            "optimizer": optim.Adam(model.parameters(), lr=lr),
            "loss_fn": nn.CrossEntropyLoss(),
            "metric_fn": accuracy,
            "metric_name": "accuracy"
        }

        return model, config
    

    def _lstm_simple(self):        
        cells = 16
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