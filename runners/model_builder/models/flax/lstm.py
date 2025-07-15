
from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype
from jax import Array


class LSTM(nn.Module):
    features: int
    return_sequences: bool = True
    return_state: bool = False
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, initial_state: Tuple[Array, Array], *, deterministic: bool = True):
        """ 
        x: (batch, time, input_dim)
        initial_state: (h, c), both (batch, features)
        Returns: 
            (batch, time, features), h, c  if return_sequences and return_state
            (batch, time, features)        if return_sequences
            (h, c)                         if not return_sequences and return_state
        """
        lstm_cell = nn.OptimizedLSTMCell(
            kernel_init=nn.initializers.xavier_uniform(),
            recurrent_kernel_init=nn.initializers.orthogonal(),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        # Apply cell to each step of the temporal window
        def step_fn(carry, x):
            h, c = carry
            (h_new, c_new), y = lstm_cell((h, c), x)
            return (h_new, c_new), y

        (last_h, last_c), outputs = nn.scan(
            fn=step_fn,
            variable_broadcast="params",
            split_rngs={'params': False},
            in_axes=1, out_axes=1,  # along temporal axis
            length=x.shape[1],
        )(initial_state, x)

        if self.return_sequences and self.return_state:
            return outputs, last_h, last_c
        elif self.return_sequences:
            return outputs
        elif self.return_state:
            return last_h, last_c
        else:
            return outputs
