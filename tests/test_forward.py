"""
Same function: with the canonical Keras weights transplanted, the native models give the
same outputs (inference, fp32) and the same loss and metric on the canonical batch.
"""

import numpy as np
import pytest

from tests.helpers import MODEL_IDS, MODELS, keras_reference, requires_flax, requires_torch
from tests.native_models import (
    flax_loss_and_metric,
    flax_outputs,
    flax_with_keras_weights,
    torch_loss_and_metric,
    torch_outputs,
    torch_with_keras_weights,
)

OUTPUT_TOLERANCE = dict(rtol=1e-4, atol=1e-5)
LOSS_TOLERANCE = dict(rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_keras_reference_is_informative(model_type, complexity):
    """Guards the other tests: constant or non-finite outputs would make any comparison pass."""
    outputs = keras_reference(model_type, complexity)["outputs"]

    assert np.all(np.isfinite(outputs))
    assert np.std(outputs, axis=0).max() > 1e-3


@requires_torch
@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_torch_outputs_match_keras(model_type, complexity):
    reference = keras_reference(model_type, complexity)
    model, _ = torch_with_keras_weights(model_type, complexity)

    outputs = torch_outputs(model, reference["x"])

    np.testing.assert_allclose(outputs, reference["outputs"], **OUTPUT_TOLERANCE)


@requires_torch
@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_torch_loss_and_metric_match_keras(model_type, complexity):
    reference = keras_reference(model_type, complexity)
    model, config = torch_with_keras_weights(model_type, complexity)

    loss, metric = torch_loss_and_metric(model_type, model, config, reference["x"], reference["y"])

    np.testing.assert_allclose(loss, reference["loss"], **LOSS_TOLERANCE, err_msg="loss")
    np.testing.assert_allclose(metric, reference["metric"], **LOSS_TOLERANCE, err_msg="metric")


@requires_flax
@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_flax_outputs_match_keras(model_type, complexity):
    reference = keras_reference(model_type, complexity)
    model, _, variables = flax_with_keras_weights(model_type, complexity)

    outputs = flax_outputs(model, variables, reference["x"])

    np.testing.assert_allclose(outputs, reference["outputs"], **OUTPUT_TOLERANCE)


@requires_flax
@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_flax_loss_and_metric_match_keras(model_type, complexity):
    reference = keras_reference(model_type, complexity)
    model, config, variables = flax_with_keras_weights(model_type, complexity)

    loss, metric = flax_loss_and_metric(model, config, variables, reference["x"], reference["y"])

    np.testing.assert_allclose(loss, reference["loss"], **LOSS_TOLERANCE, err_msg="loss")
    np.testing.assert_allclose(metric, reference["metric"], **LOSS_TOLERANCE, err_msg="metric")
