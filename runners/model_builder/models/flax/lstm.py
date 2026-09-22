
import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn, traverse_util
from flax.typing import Dtype
from jax import Array

GATES = 4 # i, f, g, o


def kernel_init(gates = GATES):
    """Glorot, as in Keras, whose kernel holds the four gates: Flax keeps one kernel per gate."""
    def init(key, shape, dtype = jnp.float32):
        limit = math.sqrt(6 / (shape[0] + gates * shape[1]))
        return jax.random.uniform(key, shape, jnp.float32, -limit, limit).astype(dtype)

    return init


def recurrent_kernel_init(gates = GATES):
    """Orthogonal in float32, since QR has no float16 or bfloat16 kernel.

    The scale is that of a slice of Keras' (h, gates * h) orthogonal matrix.
    """
    orthogonal = nn.initializers.orthogonal(scale=1 / math.sqrt(gates))

    def init(key, shape, dtype = jnp.float32):
        return orthogonal(key, shape, jnp.float32).astype(dtype)

    return init


def unit_forget_bias(params):
    """Keras' unit_forget_bias: the cell shares one bias initializer, so the forget gate is set afterwards."""
    tensors = traverse_util.flatten_dict(params)

    return traverse_util.unflatten_dict({
        path: jnp.ones_like(tensor) if path[-2:] == ("hf", "bias") else tensor
        for path, tensor in tensors.items()
    })


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
            kernel_init=kernel_init(),
            recurrent_kernel_init=recurrent_kernel_init(),
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
