"""
Shared pytest configuration: chooses the device (CPU by default, `--gpu` to use one of the
GPUs), requires KERAS_BACKEND (it decides which native framework is compared against Keras),
and resets seeds and the Keras dtype policy around every test.
"""

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def pytest_addoption(parser):
    parser.addoption(
        "--gpu",
        nargs="?",
        const="0",
        default=None,
        metavar="IDS",
        help="Run on these GPUs instead of the CPU (default: 0)",
    )


def pytest_configure(config):
    if os.environ.get("KERAS_BACKEND") not in ("tensorflow", "torch", "jax"):
        pytest.exit("Set KERAS_BACKEND to tensorflow, torch or jax (see tests/README.md)", returncode=4)

    # The frameworks read these when they are imported, which happens while collecting the tests
    gpus = config.getoption("--gpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus or ""

    if gpus:
        os.environ.pop("JAX_PLATFORMS", None)
        # The tests only check correctness: no need to hold 75% of the GPU
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"

    # Keras' numpy 2 conversion warns on every tensor and buries the report
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning:keras")


@pytest.fixture(scope="session", autouse=True)
def gpu_setup():
    """Check the GPU is really there, and disable TF32: on Ampere+ it truncates float32 matmuls to
    a 10-bit mantissa (~1e-3 of error) and would break the tolerances. The experiments keep TF32."""
    from tests.helpers import BACKEND, on_gpu

    if not on_gpu():
        return

    if BACKEND == "torch":
        import torch
        available = torch.cuda.is_available()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif BACKEND == "tensorflow":
        import tensorflow as tf
        available = bool(tf.config.list_physical_devices("GPU"))
        tf.config.experimental.enable_tensor_float_32_execution(False)
    else:
        import jax
        available = jax.default_backend() != "cpu"
        jax.config.update("jax_default_matmul_precision", "highest")

    if not available:
        pytest.exit(f"--gpu was given, but {BACKEND} sees no GPU (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})", returncode=4)


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
