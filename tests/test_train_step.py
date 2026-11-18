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
# A pre-activation within rounding of zero can fall on the other side of a ReLU in each framework:
# that moves one sample's contribution, 2.5e-4 of the layer's scale in the simple Transformer
GRADIENT_ATOL = 1e-3
STATISTICS_TOLERANCE = dict(rtol=1e-4, atol=1e-6)
# The wide ResNet flips some ReLU on almost every batch of 8, and the flip reaches every earlier layer;
# this batch flips none, and its gradients match to 1e-5
BATCHES = {("cnn", "complex"): dict(batch_size=2, seed=1)}
# On a GPU, Keras and torch pick different cuDNN algorithms and that batch flips a ReLU anyway:
# float64 leaves no rounding to flip it (the fixture restores the policy)
TORCH_FLOAT64 = {("cnn", "complex")}


def keras_before_step(model_type, complexity, dtype="float32"):
    """Fresh Keras model with the canonical weights, no dropout and SGD, plus the batch to train on."""
    import keras

    keras.config.set_dtype_policy(dtype)
    model = build_keras(model_type, complexity)
    set_canonical_weights(model)

    for layer in keras_leaves(model):
        if isinstance(layer, keras.layers.Dropout):
            layer.rate = 0.0
        elif isinstance(layer, keras.layers.MultiHeadAttention):
            # keras_leaves stops at the attention layer, which keeps its dropout as a sublayer
            layer._dropout_layer.rate = 0.0

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=1.0), loss=model.loss)

    return model, *make_batch(model_type, **BATCHES.get((model_type, complexity), {}))


def gradient_mismatches(pairs_before, pairs_after, arrays_fn, native_tensors):
    """Parameters whose gradient, or batch statistics after the step, differ from Keras'."""
    mismatches = []

    for ((kind, _, before, flatten_shape), native), ((_, _, after, _), _) in zip(pairs_before, pairs_after):
        expected_gradients = arrays_fn(kind, {name: before[name] - after[name] for name in before}, flatten_shape)
        expected_values = arrays_fn(kind, after, flatten_shape)
        tensors_before, tensors_after = native_tensors(native, before=True), native_tensors(native)

        # At the scale of the whole layer: some gradients are zero in exact arithmetic, like the attention's
        # key bias (softmax ignores a shift shared by all keys), and alone they would only compare rounding noise
        gradient_sizes = [np.abs(gradient).max() for name, gradient in expected_gradients.items() if not name.startswith("batch_stats")]
        scale = max(gradient_sizes + [1e-12])

        for name in expected_values:
            # Gradients only make sense for parameters: batch statistics follow the momentum rule
            if name.startswith("batch_stats"):
                close = np.allclose(expected_values[name], tensors_after[name], **STATISTICS_TOLERANCE)
                difference = np.abs(np.asarray(expected_values[name]) - np.asarray(tensors_after[name])).max()
            else:
                gradient = np.asarray(tensors_before[name]) - np.asarray(tensors_after[name])
                close = np.allclose(expected_gradients[name], gradient, atol=GRADIENT_ATOL * scale, **GRADIENT_TOLERANCE)
                difference = np.abs(expected_gradients[name] - gradient).max() / scale

            if not close:
                mismatches.append(f"{kind} {native[1]} {name}: difference {difference:.3g}")

    return mismatches


@requires_torch
@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_torch_train_step_matches_keras(model_type, complexity):
    import torch
    import torch.nn as nn

    from runners.torch_runner import TorchRunner
    from tests.native_models import build_torch, torch_device
    from tests.transplant import keras_to_torch, torch_arrays, torch_pairs, torch_tensors

    dtype = torch.float64 if (model_type, complexity) in TORCH_FLOAT64 else torch.float32
    keras_model, x, y = keras_before_step(model_type, complexity, str(dtype).removeprefix("torch."))
    model, config = build_torch(model_type, complexity)
    model.to(dtype)
    keras_to_torch(keras_model, model)
    model.train()

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    # Skips __init__, which pins cuda:0 and enables TF32: only the attributes the step reads
    runner = TorchRunner.__new__(TorchRunner)
    runner.model_type, runner.model, runner.amp = model_type, model, False
    runner.device, runner.dtype = torch.device(torch_device()), dtype
    runner.config = {**config, "optimizer": torch.optim.SGD(model.parameters(), lr=1.0)}
    runner.forward = runner._TorchRunner__forward

    pairs_before = torch_pairs(keras_model, model)
    numpy_tensors = lambda module: {name: tensor.detach().cpu().numpy().copy() for name, tensor in torch_tensors(module).items()}
    tensors_before = {name: numpy_tensors(module) for _, (_, name, _, module) in pairs_before}
    results = keras_model.train_on_batch(x, y, return_dict=True)

    # Samples per channel each BatchNorm sees, to undo torch's unbiased running variance below
    samples = {}
    hooks = [module.register_forward_hook(lambda module, inputs, _: samples.__setitem__(module, inputs[0].numel() // inputs[0].shape[1]))
             for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    loss, _ = runner._TorchRunner__train_step(torch.from_numpy(x), torch.from_numpy(y))
    for hook in hooks:
        hook.remove()

    np.testing.assert_allclose(float(loss), results["loss"], rtol=1e-4, atol=1e-6, err_msg="loss")

    tensors_after = {name: numpy_tensors(module) for _, (_, name, _, module) in pairs_before}
    for _, native in pairs_before:
        kind, name = native[:2]
        if kind == "lstm":
            # Both torch biases get Keras' bias gradient (the double bias, point 9): check it, then
            # leave bias_hh out, since the transplant maps Keras' bias to bias_ih alone
            ih, hh = (tensors_before[name][b] - tensors_after[name][b] for b in ("bias_ih_l0", "bias_hh_l0"))
            np.testing.assert_allclose(hh, ih, rtol=1e-5, atol=1e-7, err_msg=f"{name} bias_hh_l0")
            tensors_after[name]["bias_hh_l0"] = tensors_before[name]["bias_hh_l0"]
        elif kind == "bn":
            # torch moves running_var towards the variance times n/(n-1), Keras towards the plain one
            module, n = native[3], samples[native[3]]
            old_var, momentum = tensors_before[name]["running_var"], module.momentum
            batch_var = (old_var + (tensors_after[name]["running_var"] - old_var) / momentum) * (n - 1) / n
            tensors_after[name]["running_var"] = (1 - momentum) * old_var + momentum * batch_var

    native_tensors = lambda native, before=False: (tensors_before if before else tensors_after)[native[1]]
    pairs_after = torch_pairs(keras_model, model)
    mismatches = gradient_mismatches(pairs_before, pairs_after, torch_arrays, native_tensors)

    assert not mismatches, "\n".join(mismatches)


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
        if isinstance(context.module, nn.Dropout):
            return args[0]
        if isinstance(context.module, nn.MultiHeadDotProductAttention):
            kwargs = {**kwargs, "deterministic": True}
        return next_fun(*args, **kwargs)

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
