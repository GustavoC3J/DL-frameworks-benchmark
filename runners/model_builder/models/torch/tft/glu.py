
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    def __init__(self, hidden_units, dropout_rate=None, time_distributed=True):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.time_distributed = time_distributed

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else nn.Identity()
        self.dense = nn.Linear(hidden_units, hidden_units * 2)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        return F.glu(x)
