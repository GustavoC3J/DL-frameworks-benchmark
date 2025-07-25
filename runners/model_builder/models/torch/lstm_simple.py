
import torch.nn as nn

class LSTMSimple(nn.Module):
    def __init__(self, cells, dropout):
        super().__init__()
        
        self.lstm1 = nn.LSTM(input_size=11, hidden_size=cells, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(input_size=cells, hidden_size=cells, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        linear_size = cells // 2
        
        # Linear and output
        self.output = nn.Sequential(
            nn.Linear(cells, linear_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(linear_size, 1),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x: [batch, window, features]
        # BatchNorm1d expects [batch, features, window]

        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Take last element
        x = self.dropout2(x)
        
        # Linear and dropout
        x = self.output(x)
        return x