
from typing import Optional

import jax.numpy as jnp
from jax import Array
from flax import linen as nn
from flax.typing import Dtype

from runners.model_builder.models.flax.tft.gated_residual_network import \
    GatedResidualNetwork


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN)"""

    hidden_units: int
    num_inputs: int
    dropout_rate: Optional[float] = None
    time_distributed: bool = True
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: Array, context=None, *, deterministic: bool = True):
        # inputs: (batch_size, num_inputs, hidden_units) or (batch_size, window, num_inputs, hidden_units)

        input_shape = inputs.shape

        # Variable selection weights
        if self.time_distributed:
            flatten_inputs = inputs.reshape(input_shape[0], input_shape[1], -1)  # (batch, window, num_inputs * hidden_units)
        else:
            flatten_inputs = inputs.reshape(input_shape[0], -1)  # (batch, num_inputs * hidden_units)

        weights = GatedResidualNetwork(
            hidden_units=self.hidden_units,
            output_size=self.num_inputs,
            dropout_rate=self.dropout_rate,
            time_distributed=self.time_distributed,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(flatten_inputs, context, deterministic=deterministic)  # (batch, <window,> num_inputs)
        weights = nn.softmax(weights)  # (batch, <window,> num_inputs)
        weights = jnp.expand_dims(weights, axis=-1)  # (batch, <window,> num_inputs, 1)

        # Process each input with its corresponding GRN
        grn_outputs = []
        for i in range(self.num_inputs):
            if self.time_distributed:
                input_slice = inputs[:, :, i:i+1, :]  # (batch, window, 1, hidden_units)
            else:
                input_slice = inputs[:, i:i+1, :]      # (batch, 1, hidden_units)
            grn_outputs.append(
                GatedResidualNetwork(
                    hidden_units=self.hidden_units,
                    dropout_rate=self.dropout_rate,
                    time_distributed=self.time_distributed,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )(input_slice, context, deterministic=deterministic)
            )  # (batch, <window,> 1, hidden_units) * num_inputs

        transformed = jnp.concatenate(grn_outputs, axis=-2)  # (batch_size, <window,> num_inputs, hidden_units)

        # Multiply weights and sum
        combined = transformed * weights  # (batch, <window,> num_inputs, hidden_units)
        output = jnp.sum(combined, axis=-2)  # (batch, <window,> hidden_units)

        return output
