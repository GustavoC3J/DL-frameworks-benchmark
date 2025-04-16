

from typing import Any
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from runners.model_builder.model_builder import ModelBuilder
from runners.model_builder.models.flax.CNNSimple import CNNSimple
from runners.model_builder.models.flax.MLPSimple import MLPSimple
from runners.model_builder.models.flax.LSTMSimple import LSTMSimple


class FlaxModelBuilder(ModelBuilder):

    def __init__(self, model_type, model_complexity, key):
        super().__init__(model_type, model_complexity)

        self.key = key
    

    def _mlp_simple(self):
        lr = 1e-4

        model = MLPSimple()

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

        self.key, model_key, init_key = jax.random.split(self.key, num=3)

        model = LSTMSimple(cells, dropout, model_key)
        
        # Initial state
        dummy_input = jnp.ones((1, window, 11))
        variables = model.init(init_key, dummy_input)

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


