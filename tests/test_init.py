"""
Same initialization: each tensor a builder initializes follows the same distribution as its
Keras counterpart. Tensors are compared in the native layout (so torch's extra LSTM bias and
Flax's per-gate LSTM kernels are checked too) by mean and standard deviation.
"""

import numpy as np
import pytest

from tests.helpers import MODEL_IDS, MODELS, build_keras, make_batch, requires_flax, requires_torch
from tests.native_models import build_flax, build_torch
from tests.transplant import flax_arrays, flax_pairs, flax_tensors, torch_arrays, torch_pairs, torch_tensors


def distribution_mismatch(keras_array, native_array):
    """Why two tensors do not look drawn from the same distribution, or None.

    Tolerances grow for small tensors, whose statistics are noisy.
    """
    a = np.asarray(keras_array, dtype="float64").ravel()
    b = np.asarray(native_array, dtype="float64").ravel()
    std_a, std_b = a.std(), b.std()
    describe = f"Keras mean {a.mean():.3g} std {std_a:.3g} vs native mean {b.mean():.3g} std {std_b:.3g}"

    if std_a < 1e-12 and std_b < 1e-12:
        return None if np.isclose(a.mean(), b.mean(), atol=1e-6) else describe
    if std_a < 1e-12 or std_b < 1e-12:
        return describe

    std_tolerance = max(0.1, 5 / np.sqrt(2 * a.size))
    mean_tolerance = 5 * max(std_a, std_b) / np.sqrt(a.size)

    if abs(std_b / std_a - 1) > std_tolerance or abs(a.mean() - b.mean()) > mean_tolerance:
        return describe

    return None


def init_mismatches(pairs, arrays_fn, native_tensors):
    mismatches = []

    for (kind, _, keras_arrays, flatten_shape), native in pairs:
        tensors = native_tensors(native)
        for name, expected in arrays_fn(kind, keras_arrays, flatten_shape).items():
            problem = distribution_mismatch(expected, tensors[name])
            if problem:
                mismatches.append(f"{kind} {native[1]} {name}: {problem}")

    return mismatches


@requires_torch
@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_torch_initialization_matches_keras(model_type, complexity):
    keras_model = build_keras(model_type, complexity)
    torch_model, _ = build_torch(model_type, complexity)

    native_tensors = lambda native: {name: t.detach().cpu().numpy() for name, t in torch_tensors(native[3]).items()}
    mismatches = init_mismatches(torch_pairs(keras_model, torch_model), torch_arrays, native_tensors)

    assert not mismatches, "\n".join(mismatches)


@requires_flax
@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_flax_initialization_matches_keras(model_type, complexity):
    keras_model = build_keras(model_type, complexity)
    model, _, variables = build_flax(model_type, complexity)
    x, _ = make_batch(model_type, batch_size=1)

    native_tensors = lambda native: flax_tensors(variables, native[3])
    mismatches = init_mismatches(flax_pairs(keras_model, model, variables, x), flax_arrays, native_tensors)

    assert not mismatches, "\n".join(mismatches)
