# Benchmarking the Deep Learning Stack

This project evaluates the performance and efficiency of different frameworks and training libraries for **Deep Learning**.

## Features

- Comparison of multiple frameworks such as TensorFlow, PyTorch, and JAX, with and without Keras.
- Evaluation of MLP, CNN, and LSTM models over two variants: simple and complex.
- Selection of the precision format: fp32, fp16, bf16, mixed_fp16, mixed_bf16.
- Assessment of key metrics such as training time and energy consumption.

## Environment setup

To run this project, it is recommended to use **Miniconda** to manage the environments. Follow these steps:

### Install Miniconda

Download and install Miniconda from the official website: [Download Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)

### Create the working environments

This project uses three Conda environments, each defined in `.yml` files located in the `environments/` folder. To create them, run the following commands from the project's root directory:

```sh
conda env create -f environments/tf_env.yml
conda env create -f environments/torch_env.yml
conda env create -f environments/jax_env.yml
```

### Download the datasets

Inside the `datasets/` folder you will find a Python script to download the datasets. Simply run:

```sh
python datasets/download.py
```


## Running an experiment

Inside the `bash/` folder you will find the script `run.sh`. This script receives the experiment parameters, activates the corresponding conda environment, and executes the Python script `experiment.run`.

The parameters are the following:
  1. Framework: tf-keras, torch, torch-keras, jax, jax-keras.
  2. Model: mlp, cnn, lstm.
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
