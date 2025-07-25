import flax.linen as nn
import jax.numpy as jnp


class LSTMComplex(nn.Module):
    lstm_layers: int
    initial_cells: int
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
        
        scanLSTM = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1
        )
        
        batch_size = x.shape[0]
        current_cells = self.initial_cells

        for i in range(1, self.lstm_layers + 1):
            # From the middle, the cells are halved
            if i > (self.lstm_layers // 2):
                current_cells = current_cells // 2

            lstm = scanLSTM(current_cells, dtype=self.dtype, param_dtype=self.param_dtype)
            carry = zero_carry(batch_size, current_cells)
            _, x = lstm(carry, x)
                
            x = nn.BatchNorm(axis=-1)(x, use_running_average=not training)
            x = nn.Dropout(self.dropout)(x, deterministic=not training)


        # Funnel and output layer
        for dim, drop in zip([256, 128, 64], [0.2, 0.2, 0.1]):
            x = nn.Dense(dim, dtype=self.dtype, param_dtype=self.param_dtype)(x)
            x = nn.BatchNorm()(x, use_running_average=not training)
            x = nn.tanh(x)
            x = nn.Dropout(drop)(x, deterministic=not training)

        x = nn.Dense(1, dtype=self.dtype, param_dtype=self.param_dtype)(x)

        return x