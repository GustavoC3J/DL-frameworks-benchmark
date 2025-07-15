
from typing import Optional

import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype


class GLU(nn.Module):
    hidden_units: int
    dropout_rate: Optional[float] = None
    time_distributed: bool = True
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, *, deterministic: bool = True):
        # Dropout
        x = nn.Dropout(rate=self.dropout_rate)(inputs, deterministic=deterministic) if self.dropout_rate else inputs

        # Dense layer with 2 * hidden_units
        x = nn.Dense(
            self.hidden_units * 2,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        
        return nn.glu(x)
