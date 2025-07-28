import flax.linen as nn
import jax.numpy as jnp

from runners.model_builder.models.flax.lstm import LSTM


class LSTMComplex(nn.Module):
    lstm_layers: int
    cells: int
    dropout: float
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
        
        batch_size = x.shape[0]
        cells = self.cells

        for i in range(1, self.lstm_layers + 1):

            lstm = LSTM(cells=cells, return_sequences=(i < self.lstm_layers), dtype=self.dtype, param_dtype=self.param_dtype)
            carry = zero_carry(batch_size, cells)
            x = lstm(carry, x)
                
            x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
            x = nn.Dropout(self.dropout)(x, deterministic=not training)
            
            # Cells are halved for the next layer
            cells = max(cells // 2, 64)


        # Funnel and output layer
        for units in [128, 64, 32]:
            x = nn.Dense(units, dtype=self.dtype, param_dtype=self.param_dtype)(x)
            x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
            x = nn.relu(x)
            x = nn.Dropout(self.dropout)(x, deterministic=not training)

        x = nn.Dense(1, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.relu(x)

        return x