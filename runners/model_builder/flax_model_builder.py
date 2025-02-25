

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from runners.model_builder.model_builder import ModelBuilder


class MlpSimple(nn.Module):
    
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

        x = nn.Dense(10)(x)

        return x
    

class LSTMSimple(nn.Module):
    cells: int
    dropout: str

    @nn.compact
    def __call__(self, x, deterministic=False):

        x = nn.RNN(nn.OptimizedLSTMCell(self.cells))(x)
        x = nn.BatchNorm(use_running_average=deterministic)(x)
        x = nn.Dropout(self.dropout, deterministic=deterministic)(x)

        x = nn.RNN(nn.OptimizedLSTMCell(self.cells))(x)
        x = nn.BatchNorm(use_running_average=deterministic)(x)
        x = nn.Dropout(self.dropout, deterministic=deterministic)(x)

        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout, deterministic=deterministic)(x)

        # Output (trip count)
        x = nn.Dense(1)(x)

        return x


class FlaxModelBuilder(ModelBuilder):

    def __init__(self, model_type, model_complexity, key):
        super().__init__(model_type, model_complexity)

        self.key = key
    

    def _mlp_simple(self):
        lr = 1e-4

        model = MlpSimple()

        # Initial state
        self.key, subkey = jax.random.split(self.key)
        dummy_input = jnp.ones((1, 784))
        params = model.init(subkey, dummy_input)['params']

        # Optimizer
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)

        config = {
            "params": params,
            "optimizer": optimizer,
            "opt_state": opt_state,
            "metric_name": "accuracy"
        }

        return model, config
    

    def _mlp_complex(self):
        raise NotImplementedError()


    def _cnn_simple(self):
        lr = 1e-4

        model = CNNSimple()

        # Initial state
        self.key, subkey = jax.random.split(self.key)
        dummy_input = jnp.ones((1, 32, 32, 3))
        params = model.init(subkey, dummy_input)['params']

        # Optimizer
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)

        config = {
            "params": params,
            "optimizer": optimizer,
            "opt_state": opt_state,
            "metric_name": "accuracy"
        }

        return model, config
    

    def _cnn_complex(self):
        raise NotImplementedError()
    

    def _lstm_simple(self):
        interval = 10
        window = 48 * 60 // interval
        
        cells = 32
        dropout = 0.1
        lr = 1e-4

        model = LSTMSimple(cells, dropout)
        
        # Initial state
        self.key, subkey = jax.random.split(self.key)
        dummy_input = jnp.ones((1, window, 11))
        variables = model.init(subkey, dummy_input)

        params = variables['params']
        batch_stats = variables['batch_stats']

        # Optimizer
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)

        config = {
            "params": params,
            "optimizer": optimizer,
            "opt_state": opt_state,
            "batch_stats": batch_stats,
            "metric_name": "mae"
        }

        return model, config

    def _lstm_complex(self):
        raise NotImplementedError()


