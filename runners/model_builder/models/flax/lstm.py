
from typing import Optional, Tuple

import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype
from jax import Array


class LSTM(nn.Module):
    features: int
    return_sequences: bool = True
    return_state: bool = False
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, initial_state: Tuple[Array, Array]):
        """ 
        x: (batch, time, input_dim)
        initial_state: (h, c), both (batch, features)
        Returns: 
            (batch, time, features), h, c  if return_sequences and return_state
            (batch, time, features)        if return_sequences
            (h, c)                         if not return_sequences and return_state
        """
        lstm_cell = nn.OptimizedLSTMCell(
            self.features,
            kernel_init=nn.initializers.xavier_uniform(),
            recurrent_kernel_init=nn.initializers.orthogonal(),
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )

        # Apply cell to each step of the temporal window        
        def body_fn(cell, carry, x):
            carry, y = cell(carry, x)
            return carry, y

        (last_h, last_c), outputs = nn.scan(
            body_fn,
            variable_broadcast="params",
            split_rngs={'params': False},
            in_axes=1, out_axes=1,  # along temporal axis
            length=x.shape[1],
        )(lstm_cell, initial_state, x)

        # Return only the last temporal output if not returning sequences
        if not self.return_sequences:
            outputs = outputs[:, -1, :]

        return (outputs, last_h, last_c) if self.return_state else outputs
