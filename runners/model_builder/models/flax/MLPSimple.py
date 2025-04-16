
import flax.linen as nn

class MLPSimple(nn.Module):
    
    @nn.compact
    def __call__(self, x, deterministic=False):
        dropout = 0.2

        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dropout(dropout, deterministic=deterministic)(x)
        
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dropout(dropout, deterministic=deterministic)(x)

        x = nn.Dense(10)(x) # softmax is applied in loss function

        return x