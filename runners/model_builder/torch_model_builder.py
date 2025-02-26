
import torch.nn as nn
import torch.optim as optim


from runners.model_builder.model_builder import ModelBuilder
from utils.torch_utils import accuracy, mae


class LSTMSimple(nn.Module):
    def __init__(self, window, cells, dropout):
        super(LSTMSimple, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=11, hidden_size=cells, batch_first=True)
        self.batchnorm1 = nn.BatchNorm1d(window)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(input_size=cells, hidden_size=cells, batch_first=True)
        self.batchnorm2 = nn.BatchNorm1d(window)
        self.dropout2 = nn.Dropout(dropout)
        
        self.linear3 = nn.Linear(cells, 16)
        self.dropout3 = nn.Dropout(dropout)
        
        self.output = nn.Linear(16, 1)
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        
        x = x[:, -1, :]  # Take last element
        x = self.linear3(x)
        x = self.dropout3(x)
        
        x = self.output(x)
        return x


class TorchModelBuilder(ModelBuilder):

    def _mlp_simple(self):
        activation = nn.ReLU()
        dropout = 0.2
        lr = 1e-4
        
        model = nn.Sequential(
            nn.Linear(784, 256),
            activation,
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            activation,
            nn.Dropout(dropout),

            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

        config = {
            "optimizer": optim.Adam(model.parameters(), lr=lr),
            "loss_fn": nn.CrossEntropyLoss(),
            "metric_fn": accuracy,
            "metric_name": "accuracy"
        }

        return model, config
    

    def _mlp_complex(self):
        raise NotImplementedError()


    def _cnn_simple(self):
        
        activation = nn.ReLU()
        dropout = 0.2
        lr = 1e-4

        model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same'),
            activation,
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            activation,
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),

            nn.Flatten(),

            # 32 * 8 * 8 is the output size after convolutions and max pooling
            nn.Linear(32 * 8 * 8, 128),
            activation,
            nn.Dropout(dropout),
            
            nn.Linear(128, 128),
            activation,
            nn.Dropout(dropout),
            
            # Output layer
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )
        
        config = {
            "optimizer": optim.Adam(model.parameters(), lr=lr),
            "loss_fn": nn.CrossEntropyLoss(),
            "metric_fn": accuracy,
            "metric_name": "accuracy"
        }

        return model, config
        


    def _cnn_complex(self):
        raise NotImplementedError()
    

    def _lstm_simple(self):
        interval = 10
        window = 48 * 60 // interval
        
        cells = 32
        dropout = 0.1
        lr = 1e-4

        # It is needed a module for LSTMs
        model = LSTMSimple(window, cells, dropout)

        config = {
            "optimizer": optim.Adam(model.parameters(), lr=lr),
            "loss_fn": nn.MSELoss(),
            "metric_fn": mae,
            "metric_name": "mae"
        }

        return model, config

    def _lstm_complex(self):
        raise NotImplementedError()