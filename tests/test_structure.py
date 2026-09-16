"""
Same architecture: layers of each kind in the same order, same shapes (modulo each
framework's conventions), same number of parameters and same output shape.
"""

import numpy as np
import pytest

from tests.helpers import MODEL_IDS, MODELS, build_keras, make_batch, requires_flax, requires_torch
from tests.native_models import build_flax, build_torch, to_torch
from tests.transplant import compare_shapes, flax_arrays, flax_pairs, torch_arrays, torch_pairs


def keras_counts(model):
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    non_trainable = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    return trainable, non_trainable


@requires_torch
@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_torch_layers_match_keras(model_type, complexity):
    keras_model = build_keras(model_type, complexity)
    torch_model, _ = build_torch(model_type, complexity)

    mismatches = compare_shapes(torch_pairs(keras_model, torch_model), torch_arrays)

    assert not mismatches, "\n".join(mismatches)


@requires_torch
@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_torch_parameter_count_matches_keras(model_type, complexity):
    import torch

    keras_model = build_keras(model_type, complexity)
    torch_model, _ = build_torch(model_type, complexity)

    # torch.nn.LSTM keeps a second bias (bias_hh) that Keras does not have
    extra_lstm_bias = sum(p.numel() for name, p in torch_model.named_parameters() if name.split(".")[-1].startswith("bias_hh"))
    trainable = sum(p.numel() for p in torch_model.parameters()) - extra_lstm_bias
    non_trainable = sum(b.numel() for name, b in torch_model.named_buffers() if not name.endswith("num_batches_tracked"))

    assert (trainable, non_trainable) == keras_counts(keras_model)

    x, _ = make_batch(model_type)
    with torch.no_grad():
        assert tuple(torch_model.eval()(to_torch(x)).shape) == tuple(keras_model(x).shape)


@requires_flax
@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_flax_layers_match_keras(model_type, complexity):
    keras_model = build_keras(model_type, complexity)
    model, _, variables = build_flax(model_type, complexity)
    x, _ = make_batch(model_type, batch_size=1)

    mismatches = compare_shapes(flax_pairs(keras_model, model, variables, x), flax_arrays)

    assert not mismatches, "\n".join(mismatches)


@requires_flax
@pytest.mark.parametrize(("model_type", "complexity"), MODELS, ids=MODEL_IDS)
def test_flax_parameter_count_matches_keras(model_type, complexity):
    import jax

    keras_model = build_keras(model_type, complexity)
    model, _, variables = build_flax(model_type, complexity)

    count = lambda tree: sum(int(np.prod(a.shape)) for a in jax.tree_util.tree_leaves(tree))
    trainable = count(variables["params"])
    non_trainable = count(variables.get("batch_stats", {}))

    assert (trainable, non_trainable) == keras_counts(keras_model)

    x, _ = make_batch(model_type)
    assert tuple(model.apply(variables, x, training=False).shape) == tuple(keras_model(x).shape)
