"""
Same training step: starting from the canonical weights, one step on the canonical batch gives
the same loss, the same gradients and the same batch statistics.

The step uses plain SGD with learning rate 1, so the weight change is the gradient itself: the
optimizers are compared in test_hyperparams.py, and Adam would mix its own differences in here.
Dropout is switched off, so the step does not depend on each framework's random stream.
"""

import numpy as np
import pytest

from tests.helpers import MODEL_IDS, MODELS, build_keras, make_batch, requires_flax, requires_torch, set_canonical_weights
from tests.native_models import build_flax
from tests.transplant import flax_arrays, flax_pairs, flax_tensors, keras_leaves, keras_to_flax

GRADIENT_TOLERANCE = dict(rtol=1e-3)
STATISTICS_TOLERANCE = dict(rtol=1e-4, atol=1e-6)


def keras_before_step(model_type, complexity):
    """Fresh Keras model with the canonical weights, no dropout and SGD, plus the batch to train on."""
    import keras

    model = build_keras(model_type, complexity)
    set_canonical_weights(model)

    for layer in keras_leaves(model):
        if isinstance(layer, keras.layers.Dropout):
            layer.rate = 0.0

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=1.0), loss=model.loss)

    return model, *make_batch(model_type)


def gradient_mismatches(pairs_before, pairs_after, arrays_fn, native_tensors):
    """Parameters whose gradient, or batch statistics after the step, differ from Keras'."""
    mismatches = []

    for ((kind, _, before, flatten_shape), native), ((_, _, after, _), _) in zip(pairs_before, pairs_after):
        expected_gradients = arrays_fn(kind, {name: before[name] - after[name] for name in before}, flatten_shape)
        expected_values = arrays_fn(kind, after, flatten_shape)
        tensors_before, tensors_after = native_tensors(native, before=True), native_tensors(native)

        for name in expected_values:
            # Gradients only make sense for parameters: batch statistics follow the momentum rule
            if name.startswith("batch_stats"):
                close = np.allclose(expected_values[name], tensors_after[name], **STATISTICS_TOLERANCE)
                difference = np.abs(np.asarray(expected_values[name]) - np.asarray(tensors_after[name])).max()
            else:
                gradient = np.asarray(tensors_before[name]) - np.asarray(tensors_after[name])
                scale = max(np.abs(expected_gradients[name]).max(), 1e-12)
                close = np.allclose(expected_gradients[name], gradient, atol=1e-5 * scale, **GRADIENT_TOLERANCE)
                difference = np.abs(expected_gradients[name] - gradient).max() / scale

            if not close:
                mismatches.append(f"{kind} {native[1]} {name}: difference {difference:.3g}")

    return mismatches


@requires_torch
@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_torch_train_step_matches_keras(model_type, complexity):
    pytest.skip("TorchRunner's training step lives inside its training loop and needs a GPU: pending the runner split")


@requires_flax
@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_flax_train_step_matches_keras(model_type, complexity):
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import optax

    from utils.jax_utils import TrainState, make_train_step

    keras_model, x, y = keras_before_step(model_type, complexity)
    model, config, variables = build_flax(model_type, complexity)
    variables = keras_to_flax(keras_model, model, variables, x[:1])

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optax.sgd(learning_rate=1.0),
        batch_stats=variables.get("batch_stats", None),
        loss_scale=None,
    )

    def without_dropout(next_fun, args, kwargs, context):
        return args[0] if isinstance(context.module, nn.Dropout) else next_fun(*args, **kwargs)

    pairs_before = flax_pairs(keras_model, model, variables, x[:1])
    results = keras_model.train_on_batch(x, y, return_dict=True)

    with nn.intercept_methods(without_dropout):
        state, loss, _ = make_train_step(config["loss_fn"], config["metric_fn"])(state, (jnp.array(x), jnp.array(y)), jax.random.key(0))

    updated = {"params": state.params}
    if state.batch_stats is not None:
        updated["batch_stats"] = state.batch_stats

    np.testing.assert_allclose(float(loss), results["loss"], rtol=1e-4, atol=1e-6, err_msg="loss")

    native_tensors = lambda native, before=False: flax_tensors(variables if before else updated, native[3])
    pairs_after = flax_pairs(keras_model, model, updated, x[:1])
    mismatches = gradient_mismatches(pairs_before, pairs_after, flax_arrays, native_tensors)

    assert not mismatches, "\n".join(mismatches)
