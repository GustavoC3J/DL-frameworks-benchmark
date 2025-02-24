

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from runners.model_builder.model_builder import ModelBuilder
from utils.jax_utils import softmax_cross_entropy, accuracy


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

class FlaxModelBuilder(ModelBuilder):

    def __init__(self, model_type, model_complexity, key):
        super().__init__(model_type, model_complexity)

        self.key = key
    

    def _mlp_simple(self):
        lr = 1e-4

        model = MlpSimple()

        dummy_input = jnp.ones((1, 784))  # Simulación de entrada para inicialización
        self.key, subkey = jax.random.split(self.key)
        params = model.init(subkey, dummy_input)['params']

        # Configuración del optimizador
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
        raise NotImplementedError()
        """
        activation = "relu"
        dropout = 0.2
        lr = 1e-4
        
        model = Sequential([
            Input(shape=(32, 32, 3)),
            Conv2D(filters=32, kernel_size=(3, 3), activation=activation, padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout),
            
            Conv2D(filters=32, kernel_size=(3, 3), activation=activation, padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout),
        
            Flatten(),
        
            Dense(128, activation=activation),
            Dropout(dropout),
        
            Dense(128, activation=activation),
            Dropout(dropout),
        
            # Output layer
            Dense(20, activation = "softmax")
        ])
        
        # Compile the model
        model.compile(
            optimizer = Adam(learning_rate = lr),             
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
        )

        return model
        """
    

    def _cnn_complex(self):
        raise NotImplementedError()
    

    def _lstm_simple(self):
        raise NotImplementedError()
        """
        interval = 10
        window = 48 * 60 // interval
        
        cells = 32
        activation = "tanh"
        dropout = 0.1
        lr = 1e-4
        
        # Build the model
        model = Sequential([
            Input(shape=(window, 11)),

            LSTM(cells, activation=activation, return_sequences=True),
            BatchNormalization(),
            Dropout(dropout),

            LSTM(cells, activation=activation),
            BatchNormalization(),
            Dropout(dropout),

            Dense(16, activation = activation),
            Dropout(dropout),

            Dense(1) # Output (trip count)
        ])

        # Compile the model
        model.compile(
            optimizer = Adam(learning_rate = lr),             
            loss = 'mse',
            metrics = ['mae']
        )
        
        return model
        """

    def _lstm_complex(self):
        raise NotImplementedError()


