"""
Keras -> torch and Keras -> Flax weight transplant.

Parametric layers are paired by kind (dense, conv, bn, ln, lstm) and in definition order,
which is the same in the three frameworks. Each Keras layer is converted to the arrays the
native layer should hold, so the same conversion serves the structure tests (shapes) and the
forward tests (values).
"""

import numpy as np

from tests.helpers import to_numpy

KINDS = ("dense", "conv", "bn", "ln", "lstm")


# --- Keras side ----------------------------------------------------------------------------

def keras_leaves(layer):
    """Leaf layers in definition order (Keras' _flatten_layers reverses the children)."""
    import keras

    for child in layer._layers:
        if isinstance(child, keras.layers.LSTM) or not child._layers:
            yield child
        else:
            yield from keras_leaves(child)


def keras_parametric_layers(model):
    """(kind, layer, arrays, flatten_shape) for every parametric Keras layer.

    flatten_shape is the NHWC shape flattened right before a dense layer, or None.
    """
    import keras

    kinds = {
        keras.layers.Dense: "dense",
        keras.layers.Conv2D: "conv",
        keras.layers.BatchNormalization: "bn",
        keras.layers.LayerNormalization: "ln",
        keras.layers.LSTM: "lstm",
    }

    layers = []
    last_conv_channels, flatten_shape = None, None

    for layer in keras_leaves(model):
        if isinstance(layer, keras.layers.Flatten) and last_conv_channels:
            flatten_shape = last_conv_channels

        kind = kinds.get(type(layer))
        if kind is None:
            continue

        if kind == "lstm":
            arrays = {name: to_numpy(getattr(layer.cell, name)) for name in ("kernel", "recurrent_kernel", "bias")}
        else:
            arrays = {variable.path.split("/")[-1]: to_numpy(variable) for variable in layer.weights}

        shape = None
        if kind == "dense" and flatten_shape:
            # Square feature maps: H = W = sqrt(features / channels)
            side = int(round(np.sqrt(arrays["kernel"].shape[0] / flatten_shape)))
            shape = (side, side, flatten_shape)
            flatten_shape = None

        if kind == "conv":
            last_conv_channels = arrays["kernel"].shape[-1]

        layers.append((kind, layer, arrays, shape))

    return layers


def pair_by_kind(keras_layers, native_layers):
    """Zip Keras and native layers of each kind in order. Raises if the counts differ."""
    pairs = []

    for kind in KINDS:
        k = [layer for layer in keras_layers if layer[0] == kind]
        n = [layer for layer in native_layers if layer[0] == kind]
        assert len(k) == len(n), f"{kind}: {len(k)} layers in Keras, {len(n)} in the native model"
        pairs += list(zip(k, n))

    return pairs


def compare_shapes(pairs, native_arrays_fn):
    """Mismatches between the arrays each native layer holds and those converted from Keras."""
    mismatches = []

    for (kind, _, keras_arrays, flatten_shape), native in pairs:
        expected = {name: array.shape for name, array in native_arrays_fn(kind, keras_arrays, flatten_shape).items()}
        actual = native_shapes(native)

        if expected != actual:
            mismatches.append(f"{kind} {native[1]}: expected {expected}, got {actual}")

    return mismatches


def native_shapes(native):
    return {name: tuple(shape) for name, shape in native[2].items()}


# --- torch ----------------------------------------------------------------------------------

def torch_tensors(module):
    """Parameters and buffers of a module, by name."""
    tensors = dict(module.named_parameters(recurse=False))
    tensors.update({n: b for n, b in module.named_buffers(recurse=False) if n != "num_batches_tracked"})
    return tensors


def torch_parametric_modules(model):
    """(kind, name, {tensor name: shape}, module) in registration order."""
    import torch.nn as nn

    kinds = {nn.Linear: "dense", nn.Conv2d: "conv", nn.BatchNorm2d: "bn", nn.LayerNorm: "ln", nn.LSTM: "lstm"}
    modules = []

    for name, module in model.named_modules():
        kind = kinds.get(type(module))
        if kind is not None:
            shapes = {n: tuple(t.shape) for n, t in torch_tensors(module).items()}
            modules.append((kind, name, shapes, module))

    return modules


def torch_arrays(kind, arrays, flatten_shape=None):
    """Arrays a torch layer must hold to compute the same function as the Keras layer."""
    if kind == "dense":
        kernel = arrays["kernel"]
        if flatten_shape:
            # Keras flattens NHWC and torch NCHW: reorder the input features
            kernel = kernel.reshape(*flatten_shape, -1).transpose(2, 0, 1, 3).reshape(kernel.shape)
        return {"weight": kernel.T, "bias": arrays["bias"]}

    if kind == "conv":
        converted = {"weight": arrays["kernel"].transpose(3, 2, 0, 1)}
        if "bias" in arrays:
            converted["bias"] = arrays["bias"]
        return converted

    if kind == "bn":
        return {
            "weight": arrays["gamma"],
            "bias": arrays["beta"],
            "running_mean": arrays["moving_mean"],
            "running_var": arrays["moving_variance"],
        }

    if kind == "ln":
        return {"weight": arrays["gamma"], "bias": arrays["beta"]}

    if kind == "lstm":
        # Same gate order (i, f, g, o); torch splits the bias in two
        return {
            "weight_ih_l0": arrays["kernel"].T,
            "weight_hh_l0": arrays["recurrent_kernel"].T,
            "bias_ih_l0": arrays["bias"],
            "bias_hh_l0": np.zeros_like(arrays["bias"]),
        }

    raise ValueError(kind)


def torch_pairs(keras_model, torch_model):
    return pair_by_kind(keras_parametric_layers(keras_model), torch_parametric_modules(torch_model))


def keras_to_torch(keras_model, torch_model):
    import torch

    pairs = torch_pairs(keras_model, torch_model)
    mismatches = compare_shapes(pairs, torch_arrays)
    assert not mismatches, "\n".join(mismatches)

    with torch.no_grad():
        for (kind, _, keras_arrays, flatten_shape), (_, _, _, module) in pairs:
            tensors = torch_tensors(module)
            for name, array in torch_arrays(kind, keras_arrays, flatten_shape).items():
                tensors[name].copy_(torch.from_numpy(np.ascontiguousarray(array)))


# --- Flax -----------------------------------------------------------------------------------

def flax_called_modules(model, variables, x, types):
    """(module, path) of every module of the given types, in call order.

    Flax builds submodules while calling the model, so they can only be found by intercepting them.
    """
    import flax.linen as nn

    modules, paths = [], set()

    def interceptor(next_fun, args, kwargs, context):
        module = context.module
        if context.method_name == "__call__" and type(module) in types and module.scope.path not in paths:
            paths.add(module.scope.path)
            modules.append((module, module.scope.path))
        return next_fun(*args, **kwargs)

    with nn.intercept_methods(interceptor):
        model.apply(variables, x, training=False)

    return modules


def flax_tensors(variables, path):
    """Variables under a module path, by "collection/name"."""
    from flax import traverse_util

    tensors = {}
    for collection in variables:
        subtree = variables[collection]
        for part in path:
            subtree = subtree.get(part, {})
        for name, array in traverse_util.flatten_dict(subtree, sep="/").items():
            tensors[f"{collection}/{name}"] = array

    return tensors


def flax_parametric_modules(model, variables, x):
    """(kind, name, {"collection/name": shape}, path, module) in call order."""
    import flax.linen as nn

    from runners.model_builder.models.flax.lstm import LSTM

    kinds = {nn.Dense: "dense", nn.Conv: "conv", nn.BatchNorm: "bn", nn.LayerNorm: "ln", LSTM: "lstm"}
    modules = []

    for module, path in flax_called_modules(model, variables, x, tuple(kinds)):
        shapes = {name: tuple(array.shape) for name, array in flax_tensors(variables, path).items()}
        modules.append((kinds[type(module)], "/".join(path), shapes, path, module))

    return modules


def flax_arrays(kind, arrays, flatten_shape=None):
    """Arrays a Flax layer must hold. Flax flattens NHWC like Keras, so flatten_shape is unused."""
    if kind in ("dense", "conv"):
        converted = {"params/kernel": arrays["kernel"]}
        if "bias" in arrays:
            converted["params/bias"] = arrays["bias"]
        return converted

    if kind == "bn":
        return {
            "params/scale": arrays["gamma"],
            "params/bias": arrays["beta"],
            "batch_stats/mean": arrays["moving_mean"],
            "batch_stats/var": arrays["moving_variance"],
        }

    if kind == "ln":
        return {"params/scale": arrays["gamma"], "params/bias": arrays["beta"]}

    if kind == "lstm":
        # Keras concatenates the gates (i, f, c, o) in one kernel; Flax keeps one dense per gate
        converted = {}
        kernels = np.split(arrays["kernel"], 4, axis=1)
        recurrents = np.split(arrays["recurrent_kernel"], 4, axis=1)
        biases = np.split(arrays["bias"], 4)

        for gate, kernel, recurrent, bias in zip("ifgo", kernels, recurrents, biases):
            converted[f"params/OptimizedLSTMCell_0/i{gate}/kernel"] = kernel
            converted[f"params/OptimizedLSTMCell_0/h{gate}/kernel"] = recurrent
            converted[f"params/OptimizedLSTMCell_0/h{gate}/bias"] = bias
        return converted

    raise ValueError(kind)


def flax_pairs(keras_model, model, variables, x):
    return pair_by_kind(keras_parametric_layers(keras_model), flax_parametric_modules(model, variables, x))


def keras_to_flax(keras_model, model, variables, x):
    """New variables with the Keras weights in place of the Flax ones."""
    import jax
    import jax.numpy as jnp

    pairs = flax_pairs(keras_model, model, variables, x)
    mismatches = compare_shapes(pairs, flax_arrays)
    assert not mismatches, "\n".join(mismatches)

    new_variables = jax.tree_util.tree_map(lambda a: a, variables)

    for (kind, _, keras_arrays, flatten_shape), (_, _, _, path, _) in pairs:
        for name, array in flax_arrays(kind, keras_arrays, flatten_shape).items():
            collection, *parts = name.split("/")
            subtree = new_variables[collection]
            for part in (*path, *parts[:-1]):
                subtree = subtree[part]
            subtree[parts[-1]] = jnp.asarray(array, dtype=subtree[parts[-1]].dtype)

    return new_variables
