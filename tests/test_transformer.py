"""
Transformer-specific checks the generic equivalence tests cannot make: the decoder is causal.

The forward tests already show that torch and Flax give Keras' outputs, so checking Keras in each
of its three backends covers every framework, and it would catch a causal mask that is inverted or
silently ignored (the three use opposite conventions for it).
"""

import numpy as np
import pytest

from tests.helpers import keras_reference, to_numpy

SOURCE_LENGTH = 144
STEP = 5


@pytest.mark.parametrize("complexity", ["simple", "complex"])
def test_decoder_step_does_not_see_the_following_inputs(complexity):
    reference = keras_reference("transformer", complexity)

    # Decoder step t reads the inputs from the last source row up to t: changing those from STEP on
    # can only alter the outputs from STEP on
    changed = reference["x"].copy()
    changed[:, SOURCE_LENGTH - 1 + STEP:] += 1.0
    outputs = to_numpy(reference["model"](changed, training=False))

    np.testing.assert_allclose(outputs[:, :STEP], reference["outputs"][:, :STEP], rtol=1e-4, atol=1e-5)
    assert np.abs(outputs[:, STEP:] - reference["outputs"][:, STEP:]).max() > 1e-3
