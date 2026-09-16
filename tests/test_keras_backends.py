"""
Links the three environments: the same Keras model with the canonical weights gives the same
outputs, loss and metric on every backend. Each run leaves its results in tests/.artifacts/
and compares them with those of the backends that already ran (tests/run_all.sh runs all).
"""

import numpy as np
import pytest

from tests.helpers import BACKEND, MODEL_IDS, MODELS, keras_reference, load_other_artifacts, save_artifact


@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_keras_matches_other_backends(model_type, complexity):
    reference = keras_reference(model_type, complexity)
    save_artifact(model_type, complexity, reference)

    others = load_other_artifacts(model_type, complexity)
    if not others:
        pytest.skip("no results from other Keras backends yet: run tests/run_all.sh")

    for backend, other in others.items():
        pair = f"{BACKEND} vs {backend}"

        # Different weights would mean the variables are ordered differently between backends
        np.testing.assert_allclose(reference["checksum"], other["checksum"], rtol=1e-6, err_msg=f"{pair}: canonical weights")
        np.testing.assert_allclose(reference["outputs"], other["outputs"], rtol=1e-4, atol=1e-5, err_msg=f"{pair}: outputs")
        np.testing.assert_allclose(reference["loss"], other["loss"], rtol=1e-4, atol=1e-6, err_msg=f"{pair}: loss")
        np.testing.assert_allclose(reference["metric"], other["metric"], rtol=1e-4, atol=1e-6, err_msg=f"{pair}: metric")
