

import os
import time
from datetime import datetime

import pandas as pd

from datasets.data_loader import DataLoader
from runners.tf_runner import TFRunner
from utils.gpu_metrics import get_gpu_memory_total

# Parameters
model_type = "rnn"
model_complexity = "simple"
epochs = 3
batch_size = 64
seed = 42
n = 10

gpus = "0,1,2"

# Select which GPUs to use
os.environ["CUDA_VISIBLE_DEVICES"] = gpus


# Create output directory
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_directory = f"results/{timestamp}-{model_type}-{model_complexity}"
os.makedirs(output_directory, exist_ok=True)

global_metrics_filename = "global_metrics.csv"
results_filename = "results.csv"


# Path to the results files
global_metrics_filepath = os.path.join(output_directory, global_metrics_filename)
results_filepath = os.path.join(output_directory, results_filename)



# Perform the experiment
runner = TFRunner(
    model_type = model_type,
    model_complexity = model_complexity,
    epochs = epochs,
    batch_size=batch_size,
    seed = seed,
    n=n
)

# Define and build the model
start = time.time()
runner.define_model()
definition_time = time.time() - start

# Load formatted datasets used in training
data_loader = DataLoader(model_type, seed)
formatted_data = data_loader.load_data("train")

# Start training
start = time.time()
history = runner.train(*formatted_data)
training_time = time.time() - start

# Get memory of all GPUs
gpu_indices = [int(gpu) for gpu in gpus.split(",") if gpu.isdigit()]
gpu_memory_total = {
    f"gpu_{gpu['index']}_memory_total": gpu['memory_total'] for gpu in get_gpu_memory_total(gpu_indices)
}

global_metrics = pd.DataFrame([{  
    'definition_time': definition_time,  
    'training_time': training_time,  
    **gpu_memory_total  
}])

# Save results to csv
global_metrics.to_csv(global_metrics_filepath, index=False)
pd.DataFrame(history.history).to_csv(results_filepath, index_label="epoch")

