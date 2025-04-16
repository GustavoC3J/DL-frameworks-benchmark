
from typing import Any

import flax.linen as nn
import jax

"""
class LSTMSimple(nn.Module):
    cells: int
    dropout: str

    def setup(self):
        self.lstm1 = nn.RNN(nn.OptimizedLSTMCell(self.cells))
        self.batchnorm1 = nn.BatchNorm()
        self.dropout1 = nn.Dropout(self.dropout)
        
        self.lstm2 = nn.RNN(nn.OptimizedLSTMCell(self.cells))
        self.batchnorm2 = nn.BatchNorm()
        self.dropout2 = nn.Dropout(self.dropout)

        self.dense = nn.Dense(16)
        self.batchnorm3 = nn.BatchNorm()
        self.dropout3 = nn.Dropout(self.dropout)

        self.output = nn.Dense(1)

    def __call__(self, x, deterministic=False):

        x = self.lstm1(x)
        x = self.batchnorm1(x, use_running_average=deterministic)
        x = self.dropout1(x, deterministic=deterministic)
        
        x = self.lstm2(x)
        x = x[:, -1, :] # Keep only the last element of the window
        x = self.batchnorm2(x, use_running_average=deterministic)
        x = self.dropout2(x, deterministic=deterministic)
        
        x = self.dense(x)
        x = self.batchnorm3(x, use_running_average=deterministic)
        x = nn.tanh(x)
        x = self.dropout3(x, deterministic=deterministic)

        # Output (trip count)
        x = self.output(x)
        
        return x
"""
class LSTMSimple(nn.Module):
    cells: int
    dropout: str
    key: Any

    @nn.compact
    def __call__(self, x, deterministic=False):

        key1, key2 = jax.random.split(self.key)
        
        scanLSTM = nn.scan(
                nn.OptimizedLSTMCell, variable_broadcast="params",
                split_rngs={"params": False}, in_axes=1, out_axes=1)
        
        input_shape = x[:, 0].shape

        # First LSTM layer
        lstm = scanLSTM(self.cells)
        carry = lstm.initialize_carry(key1, input_shape)
        _, x = lstm(carry, x)
        x = nn.BatchNorm(axis=-1, use_running_average=deterministic)(x)
        x = nn.Dropout(self.dropout, deterministic=deterministic)(x)
        
        # Second LSTM layer
        lstm2 = scanLSTM(self.cells)
        carry = lstm2.initialize_carry(key2, input_shape)
        _, x = lstm2(carry, x)
        x = x[:, -1, :] # Keep only the last element of the window
        x = nn.BatchNorm(axis=-1, use_running_average=deterministic)(x)
        x = nn.Dropout(self.dropout, deterministic=deterministic)(x)

        x = nn.Dense(16)(x)
        x = nn.tanh(x)
        x = nn.Dropout(self.dropout, deterministic=deterministic)(x)

        # Output (trip count)
        x = nn.Dense(1)(x)
        
        return x

