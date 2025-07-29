
import flax.linen as nn
import jax.numpy as jnp
from runners.model_builder.models.flax.lstm import LSTM

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
        
        batch_size = x.shape[0]

        # First LSTM layer
        lstm = LSTM(self.cells, return_sequences=True, dtype=self.dtype, param_dtype=self.param_dtype)
        carry1 = zero_carry(batch_size, self.cells)
        x = lstm(x, initial_state=carry1)
        x = nn.Dropout(self.dropout)(x, deterministic=not training)
        
        # Second LSTM layer
        lstm2 = LSTM(self.cells, return_sequences=False, dtype=self.dtype, param_dtype=self.param_dtype)
        carry2 = zero_carry(batch_size, self.cells)
        x = lstm2(x, initial_state=carry2)
        x = nn.Dropout(self.dropout)(x, deterministic=not training)

        x = nn.Dense(self.cells // 2, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout)(x, deterministic=not training)

        # Output (trip count)
        x = nn.Dense(1, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.relu(x)
        
        return x

