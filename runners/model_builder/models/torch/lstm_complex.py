import torch.nn as nn

class LSTMComplex(nn.Module):
    def __init__(self, lstm_layers, initial_cells, dropout):
        super().__init__()
        
        self.lstm_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        input_size = 11  # features

        current_cells = initial_cells
        for i in range(1, lstm_layers + 1):

            # From the middle, the cells are halved
            if (i > (lstm_layers // 2)):
                current_cells = current_cells // 2

            self.lstm_layers.append(nn.LSTM(input_size, current_cells, batch_first=True))
            self.norm_layers.append(nn.BatchNorm1d(current_cells))
            self.dropout_layers.append(nn.Dropout(dropout))

            input_size = current_cells

        # Funnel and output layer
        self.linear_layers = nn.Sequential(
            nn.Linear(current_cells, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(0.1),

            nn.Linear(64, 1)
        )

        self.output_layer = nn.Linear(64, 1)

    def forward(self, x):
        # x: [batch, window, features]

        zipped_layers = zip(self.lstm_layers, self.norm_layers, self.dropout_layers)

        # Hidden LSTM layers
        for i, (lstm, norm, dropout) in enumerate(zipped_layers):
            x, _ = lstm(x)
            if i < len(self.lstm_layers) - 1:
                # BatchNorm1d expects [batch, features, window]
                x = x.transpose(1, 2)
                x = norm(x)
                x = x.transpose(1, 2)
            else:
                x = x[:, -1, :] # Take last element: [batch, features]
                x = norm(x)
            x = dropout(x)

        # Funnel and output
        x = self.linear_layers(x)

        return x