
import flax.linen as nn


class MLPComplex(nn.Module):
    dtype: any
    param_dtype: any

    dropout: float
    
    @nn.compact
    def __call__(self, x, training):
        hidden_layers = 15
        final_units = 64  # Last hidden layers will have these units
        layers_per_group = 3

        # Calculate initial units based on number of hidden layers
        groups = (hidden_layers + layers_per_group - 1) // layers_per_group
        units = final_units * (2 ** (groups - 1))  # Starting units

        # Add the hidden layers
        for i in range(1, hidden_layers + 1):
            x = nn.Dense(units, dtype=self.dtype, param_dtype=self.param_dtype)(x)
            x = nn.relu(x)
            x = nn.Dropout(self.dropout, deterministic=not training)(x)

            # Halve the number of units for the next group
            if i % layers_per_group == 0:
                units //= 2

        # Output layer
        x = nn.Dense(10, dtype=self.dtype, param_dtype=self.param_dtype)(x) # softmax is applied in loss function

        return x