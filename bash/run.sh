#!/bin/bash

# Params
BACKEND=$1  # library or library-keras
MODEL_TYPE=$2
MODEL_COMPLEXITY=$3
PRECISION=$4
GPU_IDS=$5
SEED=$6
EPOCHS=${7:-100}  # Default epochs if not provided

# Get the library
LIBRARY=$(echo "$BACKEND" | cut -d'-' -f1)

# Select conda environment
case $LIBRARY in
    tf)
        CONDA_ENV="tf_env"
        ;;
    torch)
        CONDA_ENV="torch_env"
        ;;
    jax)
        CONDA_ENV="jax_env"
        ;;
    *)
    	echo "Error: $LIBRARY"
        exit 1
        ;;
esac

# Set Keras backend (default: tf)
if [[ "$LIBRARY" != "tf" ]]; then
    export KERAS_BACKEND="$LIBRARY"
fi

# Load conda and environment 
if source "/frontend/gustcort/miniconda3/etc/profile.d/conda.sh" && conda activate $CONDA_ENV; then
    echo "Environment: $CONDA_ENV"

    # Run experiment
    python experiment.py "$BACKEND" "$MODEL_TYPE" "$MODEL_COMPLEXITY" "$PRECISION" --gpu-ids "$GPU_IDS" --seed "$SEED" --epoch $EPOCHS
else
    echo "Error: Couldn't activate conda environment: $CONDA_ENV"
    exit 1
fi