
import flax.linen as nn
import jax.numpy as jnp

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
        x = nn.Dropout(self.dropout)(x, deterministic=not training)
        
        # Second LSTM layer
        lstm2 = scanLSTM(self.cells, dtype=self.dtype, param_dtype=self.param_dtype)
        carry2 = zero_carry(batch_size, self.cells)
        _, x = lstm2(carry2, x)
        x = x[:, -1, :] # Keep only the last element of the window
        x = nn.Dropout(self.dropout)(x, deterministic=not training)

        x = nn.Dense(self.cells // 2, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout)(x, deterministic=not training)

        # Output (trip count)
        x = nn.Dense(1, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.relu(x)
        
        return x

