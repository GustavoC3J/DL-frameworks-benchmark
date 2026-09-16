"""
Shared pytest configuration: runs everything on CPU, requires KERAS_BACKEND (it decides
which native framework is compared against Keras), and resets seeds and the Keras dtype
policy around every test.
"""

import os
import sys

# Must be set before any framework is imported
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def pytest_configure(config):
    if os.environ.get("KERAS_BACKEND") not in ("tensorflow", "torch", "jax"):
        pytest.exit("Set KERAS_BACKEND to tensorflow, torch or jax (see tests/README.md)", returncode=4)

    # Keras' numpy 2 conversion warns on every tensor and buries the report
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning:keras")


@pytest.fixture(autouse=True)
def fixed_seeds_and_policy():
    import keras

    np.random.seed(0)
    keras.utils.set_random_seed(0)

    if os.environ["KERAS_BACKEND"] == "torch":
        import torch
        torch.manual_seed(0)

    yield

    keras.config.set_dtype_policy("float32")
