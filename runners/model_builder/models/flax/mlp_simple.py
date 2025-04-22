
import flax.linen as nn


class MLPSimple(nn.Module):
    dtype: any
    param_dtype: any
    
    @nn.compact
    def __call__(self, x, training):
        dropout = 0.2

        x = nn.Dense(
            256,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )(x)
        x = nn.relu(x)
        x = nn.Dropout(dropout, deterministic=not training)(x)
        
        x = nn.Dense(
            128,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )(x)
        x = nn.relu(x)
        x = nn.Dropout(dropout, deterministic=not training)(x)

        x = nn.Dense(10, dtype=self.dtype, param_dtype=self.param_dtype)(x) # softmax is applied in loss function

        return x