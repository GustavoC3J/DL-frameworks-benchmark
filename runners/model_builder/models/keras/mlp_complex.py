
import keras
from keras import layers, ops, initializers
import numpy as np

class MLPComplex(keras.Model):

    def __init__(
        self,
        groups,
        dropout,
        initial_units = 800,
        final_units = 160,
        layers_per_group = 2,
        activation = "relu",
        kernel_initializer = "glorot_uniform",
        **kwargs
    ):
        """
        groups: Number of layer groups
        activation: Activation function
        dropout: Dropout rate
        layers_per_group: Number of hidden layers per group
        initial_units: First hidden layer will have these units
        final_units: Last hidden layer will have these units
        input_dim: Number of features
        kernel_initializer: Weights initializer.
        """
        super().__init__(**kwargs)

        # Define the number of units for each group
        units_per_group = np.linspace(initial_units, final_units, groups).astype(int)
        self.model_layers = []

        # Add the hidden layers
        for units in units_per_group:
            for _ in range(layers_per_group):
                self.model_layers.append(
                    layers.Dense(
                        units,
                        activation=activation,
                        kernel_initializer=kernel_initializer,
                        bias_initializer="zeros"
                    )
                )
                self.model_layers.append(layers.Dropout(dropout))

        # Output layer
        self.model_layers.append(
            layers.Dense(
                10,
                activation="softmax",
                kernel_initializer=kernel_initializer,
                bias_initializer="zeros"
            )
        )

        # Config for model saving
        self.config = {
            "groups": groups,
            "dropout": dropout,
            "initial_units": initial_units,
            "final_units": final_units,
            "layers_per_group": layers_per_group,
            "activation": activation,
            "kernel_initializer": kernel_initializer
        }


    def call(self, inputs):
        x = inputs
        for layer in self.model_layers:
            x = layer(x)
        return x
    
    
    def get_config(self):
        super_config = super().get_config()
        super_config.update(self.config)
        return super_config



