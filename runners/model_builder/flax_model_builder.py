
import jax
import jax.numpy as jnp
import optax

from runners.model_builder.model_builder import ModelBuilder
from runners.model_builder.models.flax.cnn_complex import CNNComplex
from runners.model_builder.models.flax.cnn_simple import CNNSimple
from runners.model_builder.models.flax.lstm_simple import LSTMSimple
from runners.model_builder.models.flax.mlp_simple import MLPSimple
from runners.model_builder.models.flax.mlp_complex import MLPComplex
from runners.model_builder.models.flax.tft.tft import TFT
from utils.jax_utils import accuracy, mae, mse, softmax_cross_entropy


class FlaxModelBuilder(ModelBuilder):

    def __init__(self, model_type, model_complexity, key, policy):
        super().__init__(model_type, model_complexity)

        self.key = key

        # dtype: data type in which the calculations are performed
        # param_dtype: data type in which the parameters are stored
        self.dtype = policy.compute_dtype
        self.param_dtype = policy.param_dtype
    

    def _mlp_simple(self):
        lr = 1e-4

        model = MLPSimple(self.dtype, self.param_dtype)

        # Initial state
        self.key, subkey = jax.random.split(self.key)
        dummy_input = jnp.ones((1, 784))
        params = model.init(subkey, dummy_input, training=True)['params']

        # Optimizer
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)

        config = {
            "params": params,
            "optimizer": optimizer,
            "opt_state": opt_state,
            "loss_fn": softmax_cross_entropy,
            "metric_fn": accuracy,
            "metric_name": "accuracy"
        }

        return model, config
    

    def _mlp_complex(self):
        lr = 3e-5

        model = MLPComplex(
            groups=5,
            layers_per_group=2,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            dropout=0.2,
            initial_units=800,
            final_units=160
        )

        # Initial state
        self.key, subkey = jax.random.split(self.key)
        dummy_input = jnp.ones((1, 784))
        params = model.init(subkey, dummy_input, training=False)['params']

        # Optimizer
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)

        config = {
            "params": params,
            "optimizer": optimizer,
            "opt_state": opt_state,
            "loss_fn": softmax_cross_entropy,
            "metric_fn": accuracy,
            "metric_name": "accuracy"
        }

        return model, config


    def _cnn_simple(self):
        lr = 1e-4

        model = CNNSimple(self.dtype, self.param_dtype)

        # Initial state
        self.key, subkey = jax.random.split(self.key)
        dummy_input = jnp.ones((1, 32, 32, 3))
        params = model.init(subkey, dummy_input, training=True)['params']

        # Optimizer
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)

        config = {
            "params": params,
            "optimizer": optimizer,
            "opt_state": opt_state,
            "loss_fn": softmax_cross_entropy,
            "metric_fn": accuracy,
            "metric_name": "accuracy"
        }

        return model, config
    

    def _cnn_complex(self):
        lr = 1e-3

        model = CNNComplex(
            5, # number of blocks, 6n + 2 layers
            self.dtype,
            self.param_dtype
        )

        # Initial state
        self.key, subkey = jax.random.split(self.key)
        dummy_input = jnp.ones((1, 32, 32, 3))
        variables = model.init(subkey, dummy_input, training=True)

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
            "loss_fn": softmax_cross_entropy,
            "metric_fn": accuracy,
            "metric_name": "accuracy",
        }

        return model, config
    

    def _lstm_simple(self):
        interval = 10
        window = 48 * 60 // interval
        
        cells = 32
        dropout = 0.1
        lr = 1e-4

        self.key, init_key = jax.random.split(self.key, num=2)

        model = LSTMSimple(cells, dropout, self.dtype, self.param_dtype)
        
        # Initial state
        dummy_input = jnp.ones((1, window, 11))
        variables = model.init(init_key, dummy_input, training=True)

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
            "loss_fn": mse,
            "metric_fn": mae,
            "metric_name": "mae"
        }

        return model, config

    def _lstm_complex(self):
        interval = 10
        historical_window = 8 * 60 // interval # 8h
        prediction_window = 1 # Output timesteps

        hidden_units = 64
        output_size = 1  # Output features (trip count)
        num_attention_heads = 4
        dropout_rate = 0.2
        lr = 1e-4

        observed_idx=[10]
        unknown_idx=[i for i in range(10)]

        self.key, init_key = jax.random.split(self.key, num=2)

        model = TFT(
            hidden_units = hidden_units,
            output_size = output_size,
            num_attention_heads = num_attention_heads,
            historical_window=historical_window,
            prediction_window=prediction_window,
            observed_idx=observed_idx,
            unknown_idx=unknown_idx,
            dropout_rate = dropout_rate
        )

        # Initial state
        dummy_input = jnp.ones((1, historical_window + prediction_window, 11))
        variables = model.init(init_key, dummy_input, training=False)

        params = variables['params']

        # Optimizer
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)

        config = {
            "params": params,
            "optimizer": optimizer,
            "opt_state": opt_state,
            "loss_fn": mse,
            "metric_fn": mae,
            "metric_name": "mae"
        }

        return model, config


