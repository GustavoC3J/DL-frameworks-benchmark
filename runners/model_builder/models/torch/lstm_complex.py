import torch.nn as nn

from utils.torch_utils import init_layer_weights

class LSTMComplex(nn.Module):
    def __init__(self, cells, lstm_layers, features=11, dropout_rate=0.2):
        super().__init__()
        
        self.lstm_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        input_size = features
        for _ in range(lstm_layers):
            self.lstm_layers.append(nn.LSTM(input_size, cells, batch_first=True))
            self.norm_layers.append(nn.LayerNorm(cells))
            self.dropout_layers.append(nn.Dropout(dropout_rate))

            input_size = cells
            
            # Cells are halved for the next layer
            cells = max(cells // 2, 64)

        # Funnel and output layer
        self.dense_funnel = nn.ModuleList()
        for units in [128, 64, 32]:
            linear = nn.Linear(input_size, units)
            init_layer_weights(linear, "he_uniform")

            self.dense_funnel.append(linear)
            self.dense_funnel.append(nn.LayerNorm(units))
            self.dense_funnel.append(nn.ReLU())
            self.dense_funnel.append(nn.Dropout(dropout_rate))
            input_size = units

        self.dense_funnel.append(nn.Linear(32, 1))
        self.dense_funnel.append(nn.ReLU())


    def forward(self, x):
        # x: [batch, window, features]

        zipped_layers = zip(self.lstm_layers, self.norm_layers, self.dropout_layers)
        num_lstm_layers = len(self.lstm_layers)

        # LSTM funnel
        for i, (lstm, norm, dropout) in enumerate(zipped_layers):
            x, _ = lstm(x)

            if i == num_lstm_layers - 1:
                x = x[:, -1, :] # Take last element: [batch, features]
            
            x = norm(x)
            x = dropout(x)

        # Dense funnel and output
        for layer in self.dense_funnel:
            x = layer(x)

        return x