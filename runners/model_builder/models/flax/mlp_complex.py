
from flax import linen as nn
import numpy as np
from typing import Callable


class MLPComplex(nn.Module):
    groups: int
    dropout: float
    dtype: any
    param_dtype: any
    initial_units: int = 800
    final_units: int = 160
    layers_per_group: int = 2
    activation: Callable = nn.relu
    kernel_initializer: Callable = nn.initializers.he_uniform

    @nn.compact
    def __call__(self, x, training):

        # Define the number of units for each group
        units_per_group = np.linspace(self.initial_units, self.final_units, self.groups).astype(int)

        # Add the hidden layers
        for units in units_per_group:
            for _ in range(self.layers_per_group):
                x = nn.Dense(
                    features=units,
                    kernel_init=self.kernel_initializer(dtype=self.dtype),
                    bias_init=nn.initializers.zeros,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype
                )(x)
                x = self.activation(x)
                x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)

        # Output layer
        x = nn.Dense(
            features=10,
            kernel_init=self.kernel_initializer(dtype=self.dtype),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )(x) # softmax is applied in loss function

        return x
