
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

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
    dtype: any
    param_dtype: any

    @nn.compact
    def __call__(self, x, training):

        def zero_carry(batch_size, hidden_size):
            # estado oculto (h), estado de celda (c)
            return (
                jnp.zeros((batch_size, hidden_size), dtype=x.dtype),
                jnp.zeros((batch_size, hidden_size), dtype=x.dtype)
            )
        
        scanLSTM = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1
        )
        
        batch_size = x.shape[0]

        # First LSTM layer
        lstm = scanLSTM(self.cells, dtype=self.dtype, param_dtype=self.param_dtype)
        carry1 = zero_carry(batch_size, self.cells)
        _, x = lstm(carry1, x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.Dropout(self.dropout)(x, deterministic=not training)
        
        # Second LSTM layer
        lstm2 = scanLSTM(self.cells, dtype=self.dtype, param_dtype=self.param_dtype)
        carry2 = zero_carry(batch_size, self.cells)
        _, x = lstm2(carry2, x)
        x = x[:, -1, :] # Keep only the last element of the window
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.Dropout(self.dropout)(x, deterministic=not training)

        x = nn.Dense(16, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.tanh(x)
        x = nn.Dropout(self.dropout)(x, deterministic=not training)

        # Output (trip count)
        x = nn.Dense(1, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        
        return x

