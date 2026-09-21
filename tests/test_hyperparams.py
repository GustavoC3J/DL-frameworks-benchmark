"""
Same hyperparameters: Adam (learning rate, betas, bias correction and epsilon) as each builder
configures it, momentum and epsilon of the normalization layers, and dropout rates, attention
included (and whether its dropout mask is shared across the batch, as Flax does by default).
"""

import numpy as np
import pytest

from tests.helpers import MODEL_IDS, MODELS, build_keras, make_batch, requires_flax, requires_torch
from tests.native_models import build_flax, build_torch
from tests.transplant import flax_called_modules, keras_leaves

# Large enough for epsilon to be negligible, with sign changes so the betas matter
GRADIENTS = [
    np.array(values, dtype="float32")
    for values in ([1.0, -0.5, 0.2, -0.1], [-0.3, 0.8, 0.1, 0.5], [0.6, 0.6, -1.0, 0.2], [-1.0, 0.1, 0.4, -0.7], [0.2, -0.9, 0.3, 0.1])
]
TINY_GRADIENT = 1e-8


# --- Adam -----------------------------------------------------------------------------------

def keras_adam_trajectory(optimizer):
    import keras

    optimizer = optimizer.__class__.from_config(optimizer.get_config())
    variable = keras.Variable(np.zeros(4, dtype="float32"))
    trajectory = []

    for gradient in GRADIENTS:
        optimizer.apply_gradients([(keras.ops.convert_to_tensor(gradient), variable)])
        trajectory.append(keras.ops.convert_to_numpy(variable).copy())

    return np.array(trajectory)


def torch_adam_trajectory(optimizer):
    import torch

    parameter = torch.nn.Parameter(torch.zeros(4))
    optimizer = type(optimizer)([parameter], **optimizer.defaults)
    trajectory = []

    for gradient in GRADIENTS:
        parameter.grad = torch.from_numpy(gradient)
        optimizer.step()
        trajectory.append(parameter.detach().numpy().copy())

    return np.array(trajectory)


def optax_trajectory(transformation, gradients):
    import jax.numpy as jnp
    import optax

    params = jnp.zeros(len(gradients[0]))
    state = transformation.init(params)
    trajectory = []

    for gradient in gradients:
        updates, state = transformation.update(jnp.asarray(gradient), state, params)
        params = optax.apply_updates(params, updates)
        trajectory.append(np.asarray(params))

    return np.array(trajectory)


def optax_epsilon(transformation):
    """Epsilon recovered from the first update, lr * g / (|g| + eps), with a large and a tiny gradient."""
    first_updates = -optax_trajectory(transformation, [np.array([1.0, TINY_GRADIENT], dtype="float32")])[0]
    learning_rate = first_updates[0]
    return TINY_GRADIENT * (learning_rate / first_updates[1] - 1)


def trajectory_tolerance(optimizer):
    """Positions crossing zero inflate relative errors; a thousandth of a step is still far below a wrong lr or beta."""
    return dict(rtol=1e-4, atol=1e-3 * float(optimizer.get_config()["learning_rate"]))


@requires_torch
@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_torch_adam_matches_keras(model_type, complexity):
    keras_model = build_keras(model_type, complexity)
    _, config = build_torch(model_type, complexity)

    assert type(config["optimizer"]).__name__ == type(keras_model.optimizer).__name__
    np.testing.assert_allclose(torch_adam_trajectory(config["optimizer"]), keras_adam_trajectory(keras_model.optimizer), **trajectory_tolerance(keras_model.optimizer))


@requires_torch
@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_torch_adam_epsilon_matches_keras(model_type, complexity):
    """Keras also adds epsilon before the bias correction: equal values still differ in the first steps."""
    keras_model = build_keras(model_type, complexity)
    _, config = build_torch(model_type, complexity)

    np.testing.assert_allclose(config["optimizer"].defaults["eps"], keras_model.optimizer.epsilon, rtol=1e-6)


@requires_flax
@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_optax_adam_matches_keras(model_type, complexity):
    keras_model = build_keras(model_type, complexity)
    _, config, _ = build_flax(model_type, complexity)

    np.testing.assert_allclose(optax_trajectory(config["optimizer"], GRADIENTS), keras_adam_trajectory(keras_model.optimizer), **trajectory_tolerance(keras_model.optimizer))


@requires_flax
@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_optax_adam_epsilon_matches_keras(model_type, complexity):
    """Keras also adds epsilon before the bias correction: equal values still differ in the first steps."""
    keras_model = build_keras(model_type, complexity)
    _, config, _ = build_flax(model_type, complexity)

    np.testing.assert_allclose(optax_epsilon(config["optimizer"]), keras_model.optimizer.epsilon, rtol=1e-2)


# --- Normalization and dropout --------------------------------------------------------------

def empty_hyperparameters():
    return {"bn_momentum": [], "bn_epsilon": [], "ln_epsilon": [], "dropout_rate": [], "attention_dropout_broadcast": []}


def keras_layer_hyperparameters(model):
    import keras

    values = empty_hyperparameters()

    for layer in keras_leaves(model):
        if isinstance(layer, keras.layers.BatchNormalization):
            values["bn_momentum"].append(layer.momentum)
            values["bn_epsilon"].append(layer.epsilon)
        elif isinstance(layer, keras.layers.LayerNormalization):
            values["ln_epsilon"].append(layer.epsilon)
        elif isinstance(layer, keras.layers.Dropout):
            values["dropout_rate"].append(layer.rate)
        elif isinstance(layer, keras.layers.MultiHeadAttention):
            # Its dropout drops each attention weight on its own
            values["dropout_rate"].append(layer.dropout)
            values["attention_dropout_broadcast"].append(False)

    return values


def torch_layer_hyperparameters(model):
    import torch.nn as nn

    values = empty_hyperparameters()

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            # torch weighs the new batch with momentum; Keras weighs the running average
            values["bn_momentum"].append(1 - module.momentum)
            values["bn_epsilon"].append(module.eps)
        elif isinstance(module, nn.LayerNorm):
            values["ln_epsilon"].append(module.eps)
        elif isinstance(module, nn.Dropout):
            values["dropout_rate"].append(module.p)
        elif isinstance(module, nn.MultiheadAttention):
            values["dropout_rate"].append(module.dropout)
            values["attention_dropout_broadcast"].append(False)

    return values


def flax_layer_hyperparameters(model, variables, x):
    import flax.linen as nn

    values = empty_hyperparameters()
    types = (nn.BatchNorm, nn.LayerNorm, nn.Dropout, nn.MultiHeadDotProductAttention)

    for module, _ in flax_called_modules(model, variables, x, types):
        if isinstance(module, nn.BatchNorm):
            values["bn_momentum"].append(module.momentum)
            values["bn_epsilon"].append(module.epsilon)
        elif isinstance(module, nn.LayerNorm):
            values["ln_epsilon"].append(module.epsilon)
        elif isinstance(module, nn.MultiHeadDotProductAttention):
            values["dropout_rate"].append(module.dropout_rate)
            values["attention_dropout_broadcast"].append(module.broadcast_dropout)
        else:
            values["dropout_rate"].append(module.rate)

    return values


def summarize(values):
    """[0.99, 0.99, 0.9] -> "2×0.99, 1×0.9" """
    return ", ".join(f"{values.count(value)}×{value:g}" for value in dict.fromkeys(values)) or "none"


def hyperparameter_mismatches(native, keras):
    return [
        f"{name}: Keras {summarize(keras[name])} vs native {summarize(native[name])}"
        for name in keras
        if len(native[name]) != len(keras[name]) or not np.allclose(native[name], keras[name], rtol=1e-6, atol=0)
    ]


@requires_torch
@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_torch_layer_hyperparameters_match_keras(model_type, complexity):
    keras_model = build_keras(model_type, complexity)
    torch_model, _ = build_torch(model_type, complexity)

    mismatches = hyperparameter_mismatches(torch_layer_hyperparameters(torch_model), keras_layer_hyperparameters(keras_model))

    assert not mismatches, "\n".join(mismatches)


@requires_flax
@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_flax_layer_hyperparameters_match_keras(model_type, complexity):
    keras_model = build_keras(model_type, complexity)
    model, _, variables = build_flax(model_type, complexity)
    x, _ = make_batch(model_type, batch_size=1)

    mismatches = hyperparameter_mismatches(flax_layer_hyperparameters(model, variables, x), keras_layer_hyperparameters(keras_model))

    assert not mismatches, "\n".join(mismatches)
