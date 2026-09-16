"""
Utilities shared by the equivalence suite: backend detection and skip marks, the model
grid, the canonical batch and weights (numpy + seed, so every environment regenerates
exactly the same ones) and the artifacts that link the three Keras backends.
"""

import os
from functools import lru_cache

import numpy as np
import pytest

BACKEND = os.environ.get("KERAS_BACKEND")

# Each environment compares Keras with the native framework it serves
requires_torch = pytest.mark.skipif(BACKEND != "torch", reason="torch is compared in bm_torch_env (KERAS_BACKEND=torch)")
requires_flax = pytest.mark.skipif(BACKEND != "jax", reason="Flax is compared in bm_jax_env (KERAS_BACKEND=jax)")

MODELS = [
    ("mlp", "simple"), ("mlp", "complex"),
    ("cnn", "simple"), ("cnn", "complex"),
    ("lstm", "simple"), ("lstm", "complex"),
]
MODEL_IDS = [f"{model_type}-{complexity}" for model_type, complexity in MODELS]

INPUT_SHAPES = {"mlp": (784,), "cnn": (32, 32, 3), "lstm": (144, 11)}
NUM_CLASSES = 10

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".artifacts")


def on_gpu():
    """Whether conftest left a GPU visible (pytest --gpu)."""
    return bool(os.environ.get("CUDA_VISIBLE_DEVICES", ""))


def is_classification(model_type):
    return model_type != "lstm"


def make_batch(model_type, batch_size=8, seed=0):
    """Canonical batch with the same shapes and dtypes the data loaders produce."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(batch_size, *INPUT_SHAPES[model_type])).astype("float32")

    if is_classification(model_type):
        y = rng.integers(0, NUM_CLASSES, size=batch_size).astype("int64")
    else:
        y = rng.uniform(0, 1, size=batch_size).astype("float32")

    return x, y


def to_numpy(tensor):
    import keras
    return np.asarray(keras.ops.convert_to_numpy(tensor))


def build_keras(model_type, complexity):
    """Keras model exactly as the runners get it, built on a sample input."""
    from runners.model_builder.keras_model_builder import KerasModelBuilder

    model = KerasModelBuilder(model_type, complexity).build()
    x, _ = make_batch(model_type, batch_size=1)
    model(x, training=False)

    return model


def set_canonical_weights(model, seed=0):
    """Overwrite every Keras variable with seeded values, drawn by variable order and name.

    Positive biases keep the ReLU outputs alive, so the comparisons are not trivially zero.
    """
    for index, variable in enumerate(model.weights):
        rng = np.random.default_rng([seed, index])
        name = variable.path.split("/")[-1]
        shape = tuple(variable.shape)

        if name in ("kernel", "recurrent_kernel"):
            value = rng.normal(0, 1 / np.sqrt(np.prod(shape[:-1])), shape)
        elif name in ("gamma", "moving_variance"):
            value = rng.uniform(0.5, 1.5, shape)
        elif name == "bias" and shape == (1,):
            # Regression head followed by ReLU
            value = rng.uniform(0.5, 1.0, shape)
        elif name in ("bias", "beta"):
            value = rng.uniform(0, 0.2, shape)
        elif name == "moving_mean":
            value = rng.normal(0, 0.1, shape)
        else:
            raise ValueError(f"No canonical rule for variable {variable.path}")

        variable.assign(value.astype("float32"))


def weights_checksum(model):
    return np.array([to_numpy(variable).astype("float64").sum() for variable in model.weights])


@lru_cache(maxsize=None)
def keras_reference(model_type, complexity):
    """Keras model with the canonical weights, and its outputs, loss and metric on the canonical batch.

    Cached: the tests that use it only read from the model.
    """
    model = build_keras(model_type, complexity)
    set_canonical_weights(model)

    x, y = make_batch(model_type)
    outputs = to_numpy(model(x, training=False))
    results = model.evaluate(x, y, batch_size=len(x), verbose=0, return_dict=True)
    metric_name = "accuracy" if is_classification(model_type) else "mae"

    return {
        "model": model,
        "x": x,
        "y": y,
        "outputs": outputs,
        "loss": float(results["loss"]),
        "metric": float(results[metric_name]),
        "checksum": weights_checksum(model),
    }


def artifact_path(model_type, complexity, backend):
    return os.path.join(ARTIFACTS_DIR, f"{model_type}-{complexity}__{backend}.npz")


def save_artifact(model_type, complexity, reference):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    np.savez(
        artifact_path(model_type, complexity, BACKEND),
        outputs=reference["outputs"],
        loss=reference["loss"],
        metric=reference["metric"],
        checksum=reference["checksum"],
    )


def load_other_artifacts(model_type, complexity):
    """Artifacts left by the other Keras backends, keyed by backend."""
    artifacts = {}

    for backend in ("tensorflow", "torch", "jax"):
        path = artifact_path(model_type, complexity, backend)
        if backend != BACKEND and os.path.exists(path):
            with np.load(path) as data:
                artifacts[backend] = {key: data[key] for key in data.files}

    return artifacts
