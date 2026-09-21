# Equivalence tests

The benchmark compares five backends, so the eight models have to be **the same model** in Keras,
torch and Flax. This suite checks that: same architecture, same function, same hyperparameters,
same initialization, same dtypes in every precision, and same gradients. It uses synthetic data, so
no dataset is needed.

## Running them

```sh
bash tests/run_all.sh          # add pytest arguments: --gpu 2, -k lstm, -x...
```

That runs the suite in the three conda environments, on CPU. One environment at a time:

```sh
conda activate bm_torch_env    # bm_tf_env, bm_torch_env or bm_jax_env
export KERAS_BACKEND=torch     # tensorflow, torch or jax
python -m pytest tests/ -v
```

`--gpu <ids>` runs on those GPUs instead of the CPU (`--gpu` alone uses GPU 0). Only then are the
real kernels compared: cuDNN, the fp16/bf16 ones, and torch's `autocast`, which cannot be checked
on CPU and makes the mixed precision tests skip. On a GPU, TF32 is disabled in the three
frameworks: it would truncate float32 matmuls to a 10-bit mantissa and break the tolerances.

Each environment compares its native framework against Keras **in the same process** (`torch`
against `torch-keras`, Flax against `jax-keras`) and leaves its Keras results in
`tests/.artifacts/`, which the next environment uses to check that the three Keras backends agree.
That is how `tf-keras` gets linked to the other two without ever loading two environments at once.
Running a single environment skips that comparison.

The canonical weights and batch are generated with numpy from a seed, so every environment builds
exactly the same ones without exchanging files.

## What each file covers

| File | Checks |
|---|---|
| `test_structure.py` | Parameter count (modulo torch's second LSTM bias), same layer types in the same order, compatible shapes, output shape |
| `test_forward.py` | With the canonical weights transplanted: same outputs in inference, same loss and metric on the canonical batch, through each runner's evaluation path. Also that the ViT cuts the same patches as `keras.ops.image.extract_patches` |
| `test_keras_backends.py` | That `tf-keras`, `torch-keras` and `jax-keras` give the same outputs, loss and metric |
| `test_hyperparams.py` | Adam: trajectory (learning rate, betas, bias correction) and epsilon. Momentum and epsilon of BatchNorm and LayerNorm, and dropout rates, attention included (and whether its dropout mask is shared across the batch, as Flax does by default) |
| `test_init.py` | That every tensor a builder initializes follows the same distribution as its Keras counterpart (mean and standard deviation, with a tolerance that depends on its size) |
| `test_precision.py` | For the five precisions: dtype of the parameters, declared compute dtype, and dtype of the outputs and the loss |
| `test_train_step.py` | One SGD step with learning rate 1, which exposes the gradient: same loss, same gradients (compared at the scale of their layer) and same BatchNorm statistics |

And the scaffolding: `conftest.py` (device, seeds, dtype policy), `helpers.py` (model grid,
canonical batch and weights, artifacts), `transplant.py` (pairs layers by type and converts the
weights to each framework's layout) and `native_models.py` (builds the native models and runs them
the way the runners do).

The tests use the **builders**, not the model classes, because bugs like the ResNet stem of Flax
were in the builder arguments.

## Failing tests

The failures are real discrepancies between frameworks, not problems with the suite: differences in
default epsilons, initializers, Adam's epsilon and the dtype of the loss in reduced precision. They
are being fixed one at a time.

## Adding a model or a layer

1. Add the combination to `MODELS`, its input shape to `INPUT_SHAPES` and, if it classifies, its
   number of classes to `NUM_CLASSES`, in `helpers.py`.
2. If it brings a new layer type, add it to the `kinds` dictionaries in `transplant.py` and its
   conversion to `torch_arrays` and `flax_arrays`. Structure, forward, initialization, precision and
   gradient tests then cover it automatically. If the Keras layer has sublayers of its own, like
   `LSTM` or `MultiHeadAttention`, make `keras_leaves` stop at it so it is paired as a whole.
3. If it brings a new hyperparameter, add it to the three `*_layer_hyperparameters` functions in
   `test_hyperparams.py`.
