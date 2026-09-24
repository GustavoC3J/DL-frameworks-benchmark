
import flax.linen as nn
from runners.model_builder.models.flax.lstm import LSTM

class LSTMSimple(nn.Module):
    cells: int
    dropout: str
    dtype: any
    param_dtype: any

    @nn.compact
    def __call__(self, x, training):

        # First LSTM layer
        lstm = LSTM(self.cells, return_sequences=True, dtype=self.dtype, param_dtype=self.param_dtype)
        x = lstm(x)
        x = nn.Dropout(self.dropout)(x, deterministic=not training)
        
        # Second LSTM layer
        lstm2 = LSTM(self.cells, return_sequences=False, dtype=self.dtype, param_dtype=self.param_dtype)
        x = lstm2(x)
        x = nn.Dropout(self.dropout)(x, deterministic=not training)

        x = nn.Dense(self.cells // 2, kernel_init=nn.initializers.glorot_uniform(), dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout)(x, deterministic=not training)

        # Output (trip count)
        x = nn.Dense(1, kernel_init=nn.initializers.glorot_uniform(), dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.relu(x)
        
        return x

