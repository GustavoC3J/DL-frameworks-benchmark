# Benchmarking the Deep Learning Stack

This project evaluates the performance and efficiency of different frameworks and training libraries for **Deep Learning**.

## Features

- Comparison of multiple frameworks such as TensorFlow, PyTorch, and JAX, with and without Keras.
- Evaluation of MLP, CNN, LSTM, and Transformer models over two variants: simple and complex.
- Selection of the precision format: fp32, fp16, bf16, mixed_fp16, mixed_bf16.
- Assessment of key metrics such as training time and energy consumption.

## The Transformer

It is the encoder-decoder of [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (multi-head
attention, post-LayerNorm residual blocks, sinusoidal positions, dropout on every sublayer), with a linear
regression head, trained on the taxi data like the LSTM. Given the last 24 hours (144 steps of 10 minutes),
it predicts the trip count of the next 12 steps. The decoder is fed the observations shifted one step
(teacher forcing) under a causal mask, also in validation and test, so its test metric is not a real
12-step forecast. `simple` has 1+1 layers and `complex` 2+2 (about 0.3 M and 14.7 M parameters). Unlike
the paper, it trains with the benchmark's fixed protocol: Adam with a constant learning rate and no warm-up.

## Environment setup

To run this project, it is recommended to use **Miniconda** to manage the environments. Follow these steps:

### Install Miniconda

Download and install Miniconda from the official website: [Download Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)

### Create the working environments

This project uses three Conda environments, each defined in `.yml` files located in the `environments/` folder. To create them, run the following commands from the project's root directory:

```sh
conda env create -f environments/bm_tf_env.yml
conda env create -f environments/bm_torch_env.yml
conda env create -f environments/bm_jax_env.yml
```

### Download the datasets

Inside the `datasets/` folder you will find a Python script to download the datasets. Simply run:

```sh
python datasets/download.py
```

### Preprocess the taxi dataset

The LSTM and Transformer dataset is 7 GB of raw trip records that get cleaned and aggregated into a few thousand
10-minute intervals. That result is cached, so it is computed once instead of on every run:

```sh
python -m datasets.preprocess_taxi
```

This is optional: the loader builds the cache itself the first time it needs it.


## Running an experiment

Inside the `bash/` folder you will find the script `run.sh`. This script receives the experiment parameters, activates the corresponding conda environment, and executes the Python script `experiment.run`.

The parameters are the following:
  1. Framework: tf-keras, torch, torch-keras, jax, jax-keras.
  2. Model: mlp, cnn, lstm, transformer.
  3. Complexity: simple, complex.
  4. Precision: fp32, fp16, bf16, mixed_fp16, mixed_bf16.
  5. GPU: GPU number (0, 1, 2...). Run the command `nvidia-smi` to see each GPU's id.
  6. Seed: Any number.
  7. Epochs: Any number. Default: 100.

Example 1: To run an experiment with PyTorch, a CNN model, complex version, FP32, on GPU 0 with seed 42, use:

```sh
bash/run.sh torch cnn complex fp32 0 42
```
Example 2: To run an experiment with JAX and Keras, a MLP model, simple version, mixed precision (BF16), on GPU 2 with seed 123456 over 10 epochs, use:

```sh
bash/run.sh jax-keras mlp simple mixed_bf16 2 123456 10
```

When execution finishes, the experiment results will be stored in their corresponding folder inside `results/`.

## Checking that the models are equivalent

`tests/` holds an equivalence suite: it checks that the eight models are the same model in Keras,
torch and Flax (same architecture, same outputs with the same weights, same hyperparameters,
initialization, dtypes and gradients). See [`tests/README.md`](tests/README.md).

```sh
bash tests/run_all.sh          # add --gpu <id> to run it on a GPU
```

## Model sizes

```sh
python -m utils.count_parameters
```

prints the trainable parameters of the eight models (works in any environment).

## Analysis

[`profiling/profiling_analysis.ipynb`](profiling/profiling_analysis.ipynb) profiles the benchmark
itself: how much of the training time the GPU is actually busy, why `torch-keras` is the slowest
backend, and why its LSTM stopped being an outlier after a Keras update. See
[`profiling/`](profiling/).
