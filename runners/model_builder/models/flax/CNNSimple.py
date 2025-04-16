
import flax.linen as nn

class CNNSimple(nn.Module):

    @nn.compact
    def __call__(self, x, deterministic=False):
        dropout = 0.2

        # Conv layer 1
        x = nn.Conv(32, kernel_size=(3, 3), padding='same')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Dropout(dropout, deterministic=deterministic)(x)

        # Conv layer 2
        x = nn.Conv(32, kernel_size=(3, 3), padding='same')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Dropout(dropout, deterministic=deterministic)(x)

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # Dense layers
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dropout(dropout, deterministic=deterministic)(x)

        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dropout(dropout, deterministic=deterministic)(x)

        x = nn.Dense(10)(x) # softmax is applied in loss function

        return x
    