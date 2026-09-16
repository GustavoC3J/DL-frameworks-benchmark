"""
Same precision: for each of the five precisions, the parameters, the computation, the outputs
and the loss use the same data types in every framework. Everything is checked as the runners
set it up (Keras dtype policy, model.to(dtype) or autocast, and jmp policy).
"""

from contextlib import nullcontext

import numpy as np
import pytest

from tests.helpers import MODEL_IDS, MODELS, build_keras, make_batch, on_gpu, requires_flax, requires_torch
from tests.native_models import build_flax, build_torch, to_torch
from tests.transplant import flax_pairs, flax_tensors, torch_pairs, torch_tensors
from utils.precision import Precision, get_keras_precision, get_torch_precision

PRECISIONS = list(Precision)
CASES = [(model_type, complexity, precision) for model_type, complexity in MODELS for precision in PRECISIONS]
CASE_IDS = [f"{model}-{precision.value}" for model in MODEL_IDS for precision in PRECISIONS]


def dtype_name(dtype):
    """float16 for every framework's way of spelling it (strings, numpy dtypes, jnp types, torch dtypes)."""
    try:
        return np.dtype(dtype).name
    except TypeError:
        return str(dtype).replace("torch.", "")


def keras_model(model_type, complexity, precision):
    """Keras model under its dtype policy. conftest restores the policy after each test."""
    import keras

    keras.config.set_dtype_policy(get_keras_precision(precision))

    return build_keras(model_type, complexity)


def group_mismatches(mismatches):
    """One line per (kind, Keras dtypes, native dtypes): the models repeat the same layer dozens of times."""
    grouped = {}

    for kind, name, expected, actual in mismatches:
        grouped.setdefault((kind, expected, actual), []).append(name)

    return [
        f"{len(names)} {kind} layers ({names[0]}...): Keras {expected} vs native {actual}"
        for (kind, expected, actual), names in grouped.items()
    ]


def dtype_mismatches(pairs, native_dtypes):
    """Layers whose parameters do not have the same dtypes as their Keras counterpart."""
    mismatches = []

    for (kind, layer, _, _), native in pairs:
        weights = layer.cell.weights if kind == "lstm" else layer.weights
        expected = sorted({dtype_name(weight.dtype) for weight in weights})
        actual = sorted({dtype_name(dtype) for dtype in native_dtypes(native).values()})

        if expected != actual:
            mismatches.append((kind, native[1], str(expected), str(actual)))

    return group_mismatches(mismatches)


# --- torch ----------------------------------------------------------------------------------

def model_dtype(model):
    return next(model.parameters()).dtype


def torch_model_in_precision(model_type, complexity, precision):
    """Model as TorchRunner leaves it: parameters cast in pure precision, float32 under autocast."""
    dtype, amp_dtype = get_torch_precision(precision)
    model, config = build_torch(model_type, complexity)
    model.to(dtype=dtype)

    return model.eval(), config, amp_dtype


@requires_torch
@pytest.mark.parametrize(("model_type", "complexity", "precision"), CASES, ids=CASE_IDS)
def test_torch_parameter_dtypes_match_keras(model_type, complexity, precision):
    reference = keras_model(model_type, complexity, precision)
    model, _, _ = torch_model_in_precision(model_type, complexity, precision)

    native_dtypes = lambda native: {name: t.dtype for name, t in torch_tensors(native[3]).items()}
    mismatches = dtype_mismatches(torch_pairs(reference, model), native_dtypes)

    assert not mismatches, "\n".join(mismatches)


@requires_torch
@pytest.mark.parametrize(("model_type", "complexity", "precision"), CASES, ids=CASE_IDS)
def test_torch_output_and_loss_dtypes_match_keras(model_type, complexity, precision):
    import torch

    from utils.torch_utils import adjust_outputs

    reference = keras_model(model_type, complexity, precision)
    model, config, amp_dtype = torch_model_in_precision(model_type, complexity, precision)

    if amp_dtype is not None and not on_gpu():
        pytest.skip("autocast decides the dtype of each operation on the GPU: run with --gpu")

    x, y = make_batch(model_type, batch_size=2)
    keras_outputs = reference(x, training=False)
    keras_loss = reference.compute_loss(y=y, y_pred=keras_outputs)

    # TorchRunner casts the batch in pure precision and lets autocast do it in mixed
    batch_x = to_torch(x, dtype=None if amp_dtype else model_dtype(model))
    batch_y = to_torch(y)
    autocast = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else nullcontext()

    with torch.no_grad(), autocast:
        outputs = model(batch_x)
        if model_type == "lstm":
            outputs = adjust_outputs(outputs, batch_y)
        loss = config["loss_fn"](outputs, batch_y)

    assert dtype_name(outputs.dtype) == dtype_name(keras_outputs.dtype), "outputs"
    assert dtype_name(loss.dtype) == dtype_name(keras_loss.dtype), "loss"


# --- Flax -----------------------------------------------------------------------------------

@requires_flax
@pytest.mark.parametrize(("model_type", "complexity", "precision"), CASES, ids=CASE_IDS)
def test_flax_parameter_dtypes_match_keras(model_type, complexity, precision):
    reference = keras_model(model_type, complexity, precision)
    model, _, variables = build_flax(model_type, complexity, precision=precision)
    x, _ = make_batch(model_type, batch_size=1)

    native_dtypes = lambda native: {name: array.dtype for name, array in flax_tensors(variables, native[3]).items()}
    mismatches = dtype_mismatches(flax_pairs(reference, model, variables, x), native_dtypes)

    assert not mismatches, "\n".join(mismatches)


@requires_flax
@pytest.mark.parametrize(("model_type", "complexity", "precision"), CASES, ids=CASE_IDS)
def test_flax_compute_dtypes_match_keras(model_type, complexity, precision):
    """Flax layers built without dtype infer it from their inputs instead of following the policy."""
    reference = keras_model(model_type, complexity, precision)
    model, _, variables = build_flax(model_type, complexity, precision=precision)
    x, _ = make_batch(model_type, batch_size=1)

    mismatches = []
    for (kind, layer, _, _), (_, name, _, _, module) in flax_pairs(reference, model, variables, x):
        declared = dtype_name(module.dtype) if module.dtype is not None else "inferred from the inputs"
        if declared != dtype_name(layer.compute_dtype):
            mismatches.append((kind, name, dtype_name(layer.compute_dtype), declared))

    mismatches = group_mismatches(mismatches)

    assert not mismatches, "\n".join(mismatches)


@requires_flax
@pytest.mark.parametrize(("model_type", "complexity", "precision"), CASES, ids=CASE_IDS)
def test_flax_output_and_loss_dtypes_match_keras(model_type, complexity, precision):
    import jax.numpy as jnp

    reference = keras_model(model_type, complexity, precision)
    model, config, variables = build_flax(model_type, complexity, precision=precision)

    x, y = make_batch(model_type, batch_size=2)
    keras_outputs = reference(x, training=False)
    keras_loss = reference.compute_loss(y=y, y_pred=keras_outputs)

    outputs = model.apply(variables, x, training=False)
    loss = config["loss_fn"](outputs, jnp.array(y))

    assert dtype_name(outputs.dtype) == dtype_name(keras_outputs.dtype), "outputs"
    assert dtype_name(loss.dtype) == dtype_name(keras_loss.dtype), "loss"
