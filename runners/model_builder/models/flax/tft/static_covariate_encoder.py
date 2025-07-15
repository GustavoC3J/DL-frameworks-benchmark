
from typing import Optional

import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype
from jax import Array

from runners.model_builder.models.flax.tft.gated_residual_network import \
    GatedResidualNetwork


class StaticCovariateEncoder(nn.Module):
    """StaticCovariateEncoder"""

    hidden_dim: int
    dropout_rate: Optional[float] = None
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: Array, *, deterministic: bool = True):
        """
        inputs: Tensor of shape (batch_size, hidden_size)
        """

        c_variable_selection = GatedResidualNetwork(
            hidden_units=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(inputs, deterministic=deterministic)

        c_enrichment = GatedResidualNetwork(
            hidden_units=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(inputs, deterministic=deterministic)

        c_state_h = GatedResidualNetwork(
            hidden_units=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(inputs, deterministic=deterministic)

        c_state_c = GatedResidualNetwork(
            hidden_units=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            time_distributed=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(inputs, deterministic=deterministic)

        return (
            c_variable_selection,  # Context variable selection 
            c_enrichment,          # Context enrichment
            (c_state_h, c_state_c) # Context state (h, c)
        )
