"""Trainable parameters of the eight benchmark models, as the builders create them.

    python -m utils.count_parameters

Counted on the Keras models: torch keeps a second LSTM bias that Keras and Flax do not have.
BatchNorm's moving statistics are not parameters, so they are left out.
"""

import os
from importlib.util import find_spec

# Keras defaults to TensorFlow, which only one of the environments has: use the backend installed
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = next(name for name in ("tensorflow", "torch", "jax") if find_spec(name))

import numpy as np

from runners.model_builder.keras_model_builder import KerasModelBuilder

# Input of each model, as the data loaders produce it. The Transformer's holds the 144 source
# steps and the 12 decoder steps, which start at the last source one
INPUT_SHAPES = {"mlp": (784,), "cnn": (32, 32, 3), "lstm": (144, 11), "transformer": (144 + 12 - 1, 11)}
NAMES = {"mlp": "MLP", "cnn": "CNN", "lstm": "LSTM", "transformer": "Transformer"}
COMPLEXITIES = ("simple", "complex")

if __name__ == "__main__":

    for model_type, shape in INPUT_SHAPES.items():
        for complexity in COMPLEXITIES:
            model = KerasModelBuilder(model_type, complexity).build()
            model(np.zeros((1, *shape), dtype="float32"), training=False) # Creates the weights

            parameters = sum(int(np.prod(weight.shape)) for weight in model.trainable_weights)
            print(f"{NAMES[model_type]} {complexity}: {parameters:,}")
