
from typing import Optional

import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype

from runners.model_builder.models.flax.tft.glu import GLU


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN)"""

    hidden_units: int
    output_size: Optional[int] = None
    dropout_rate: Optional[float] = None
    time_distributed: bool = True
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, context=None, *, deterministic: bool = True):
        output_size = self.output_size or self.hidden_units

        # Residual connection
        if inputs.shape[-1] != output_size:
            residual = nn.Dense(
                features=output_size,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )(inputs)
        else:
            residual = inputs

        # First dense
        x = nn.Dense(
            features=self.hidden_units,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(inputs)

        # If context is provided, apply context transformation and add it
        if context is not None:
            if self.time_distributed:
                # Expand temporal dimension
                context = jnp.expand_dims(context, axis=1)
            context_transformed = nn.Dense(
                features=self.hidden_units,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )(context)
            x = x + context_transformed

        # Activation
        x = nn.elu(x)

        # Activation, second linear and dropout (if available)
        x = nn.Dense(
            features=self.hidden_units,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        if self.dropout_rate:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        # Gating
        x = GLU(
            hidden_units=output_size,
            dropout_rate=self.dropout_rate,
            time_distributed=self.time_distributed,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x, deterministic=deterministic)

        # Residual + layer norm
        x = x + residual
        x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)

        return x
