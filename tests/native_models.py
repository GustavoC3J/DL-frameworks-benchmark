"""
Native (torch and Flax) models built with the project's builders, optionally carrying the
canonical Keras weights, and helpers to run them the way the runners do.
"""

from functools import lru_cache

import numpy as np

from tests.helpers import keras_reference, make_batch, on_gpu
from tests.transplant import keras_to_flax, keras_to_torch
from utils.precision import Precision, get_jmp_policy


# --- torch ----------------------------------------------------------------------------------

def torch_device():
    """cuda:0 is the GPU conftest left visible, as in the runners."""
    return "cuda:0" if on_gpu() else "cpu"


def to_torch(array, dtype=None):
    import torch
    return torch.from_numpy(array).to(device=torch_device(), dtype=dtype)


def build_torch(model_type, complexity):
    from runners.model_builder.torch_model_builder import TorchModelBuilder

    model, config = TorchModelBuilder(model_type, complexity).build()

    return model.to(device=torch_device()), config


@lru_cache(maxsize=None)
def torch_with_keras_weights(model_type, complexity):
    model, config = build_torch(model_type, complexity)
    keras_to_torch(keras_reference(model_type, complexity)["model"], model)
    model.eval()
    return model, config


def torch_loss_and_metric(model_type, model, config, x, y):
    """Loss and metric computed like TorchRunner's evaluation loop (int64 labels, float32 targets)."""
    import torch

    from utils.torch_utils import adjust_outputs

    batch_y = to_torch(y)

    with torch.no_grad():
        outputs = model(to_torch(x))
        if model_type == "lstm":
            outputs = adjust_outputs(outputs, batch_y)

        loss = config["loss_fn"](outputs, batch_y)
        metric = config["metric_fn"](outputs, batch_y)

    return float(loss), float(metric)


def torch_outputs(model, x):
    """Raw model outputs: the three frameworks return logits in classification."""
    import torch

    with torch.no_grad():
        outputs = model(to_torch(x))

    return outputs.cpu().numpy()


# --- Flax -----------------------------------------------------------------------------------

def build_flax(model_type, complexity, precision=Precision.FP32, seed=0):
    import jax

    from runners.model_builder.flax_model_builder import FlaxModelBuilder

    policy, _ = get_jmp_policy(precision)
    model, config = FlaxModelBuilder(model_type, complexity, jax.random.key(seed), policy).build()

    variables = {"params": config["params"]}
    if "batch_stats" in config:
        variables["batch_stats"] = config["batch_stats"]

    return model, config, variables


@lru_cache(maxsize=None)
def flax_with_keras_weights(model_type, complexity):
    model, config, variables = build_flax(model_type, complexity)
    x, _ = make_batch(model_type, batch_size=1)
    variables = keras_to_flax(keras_reference(model_type, complexity)["model"], model, variables, x)
    return model, config, variables


def flax_state(model, config, variables):
    """TrainState as JaxRunner.define_model creates it (fp32, no loss scale)."""
    from utils.jax_utils import TrainState

    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=config["optimizer"],
        batch_stats=variables.get("batch_stats", None),
        loss_scale=None,
    )


def flax_loss_and_metric(model, config, variables, x, y):
    """Loss and metric computed with the same eval step JaxRunner uses."""
    import jax.numpy as jnp

    from utils.jax_utils import make_eval_step

    state = flax_state(model, config, variables)
    loss, metric = make_eval_step(config["loss_fn"], config["metric_fn"])(state, (jnp.array(x), jnp.array(y)))

    return float(loss), float(metric)


def flax_outputs(model, variables, x):
    return np.asarray(model.apply(variables, x, training=False))
