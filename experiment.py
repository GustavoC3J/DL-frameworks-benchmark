

import argparse
import os
import time
from datetime import datetime
import traceback

import pandas as pd

from datasets.data_loader import DataLoader
from utils.gpu_metrics import get_gpu_memory_total
from utils.precision import Precision


def parse_params():
    parser = argparse.ArgumentParser()
    
    # Required params
    parser.add_argument("backend", type=str)
    parser.add_argument("model_type", type=str)
    parser.add_argument("model_complexity", type=str)
    parser.add_argument("precision", type=Precision, choices=list(Precision))
    
    # Optional params
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-ids", type=str, default="2")

    return parser.parse_args()

def run_experiment(runner, params, output_directory):

    # Path to the results files
    global_metrics_filepath = os.path.join(output_directory, "global_metrics.csv")
    train_results_filepath = os.path.join(output_directory, "train.csv")
    train_samples_filepath = os.path.join(output_directory, "train_samples.csv")
    test_results_filepath = os.path.join(output_directory, "test.csv")
    test_samples_filepath = os.path.join(output_directory, "test_samples.csv")



    # Perform the experiment
    
    # Define and build the model
    start = time.time()
    runner.define_model()
    definition_time = time.time() - start

    # Load formatted datasets used in training
    data_loader = DataLoader(params.model_type, params.seed)
    formatted_data = data_loader.load_data("train")

    # Start training
    start = time.time()
    train_results, train_samples = runner.train(*formatted_data, output_directory)
    training_time = time.time() - start

    # Start testing
    formatted_data = data_loader.load_data("test")
    start = time.time()
    test_results, test_samples = runner.evaluate(*formatted_data)
    testing_time = time.time() - start

    # Get memory of all GPUs
    gpu_indices = [int(gpu) for gpu in params.gpu_ids.split(",") if gpu.isdigit()]
    gpu_memory_total = {
        f"gpu_{gpu['index']}_memory_total": gpu['memory_total'] for gpu in get_gpu_memory_total(gpu_indices)
    }

    global_metrics = pd.DataFrame([{  
        'backend': params.backend,
        'model_type': params.model_type,
        'model_complexity': params.model_complexity,
        'precision': params.precision,
        'epochs': params.epochs,
        'batch_size': params.batch_size,
        'seed': params.seed,
        'gpu_ids': params.gpu_ids,
        'definition_time': definition_time,  
        'training_time': training_time,
        'testing_time': testing_time,  
        **gpu_memory_total  
    }])

    # Save results to csv
    global_metrics.to_csv(global_metrics_filepath, index=False)

    pd.DataFrame(train_results).to_csv(train_results_filepath, index_label="epoch")
    pd.DataFrame(train_samples).to_csv(train_samples_filepath, index=False)

    pd.DataFrame([test_results]).to_csv(test_results_filepath, index=False)
    pd.DataFrame(test_samples).to_csv(test_samples_filepath, index=False)




if __name__ == "__main__":
    params = parse_params()

    # Set GPUs to use
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_ids
    
    # backend is the library used follow by "-keras" if Keras is used
    segments = params.backend.split("-", 1)

    library = segments[0]
    use_keras = len(segments) > 1 and segments[1] == "keras"

    if (use_keras):
        import keras
        keras.utils.set_random_seed(params.seed)

    # Load the corresponding runner
    if library == "tf":
        from runners.tf_runner import TFRunner

        runner = TFRunner(
            model_type = params.model_type,
            model_complexity = params.model_complexity,
            keras = use_keras,
            epochs = params.epochs,
            batch_size=params.batch_size,
            seed = params.seed,
            gpu_ids = params.gpu_ids,
            precision = params.precision
        )

    elif library == "torch":
        from runners.torch_runner import TorchRunner

        runner = TorchRunner(
            model_type = params.model_type,
            model_complexity = params.model_complexity,
            keras = use_keras,
            epochs = params.epochs,
            batch_size=params.batch_size,
            seed = params.seed,
            gpu_ids = params.gpu_ids,
            precision = params.precision
        )
        
    elif library == "jax":
        from runners.jax_runner import JaxRunner

        runner = JaxRunner(
            model_type = params.model_type,
            model_complexity = params.model_complexity,
            keras = use_keras,
            epochs = params.epochs,
            batch_size=params.batch_size,
            seed = params.seed,
            gpu_ids = params.gpu_ids,
            precision = params.precision
        )
    else:
        print("Error: Unknown library")

    if runner:
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_directory = f"results/{timestamp}_{params.backend}_{params.model_type}_{params.model_complexity}_{params.precision}_{params.seed}"
        os.makedirs(output_directory, exist_ok=True)

        try:
            run_experiment(runner, params, output_directory)
        except Exception as e:
            # Clean traceback routes and save to file
            tb_lines = traceback.format_exc().splitlines()
            cleaned_lines = []
            
            for line in tb_lines:
                if line.strip().startswith('File'):
                    parts = line.split('"')
                    if len(parts) >= 3:
                        filename = os.path.basename(parts[1])
                        line = line.replace(parts[1], filename)
                cleaned_lines.append(line)
                
            with open(output_directory + "/error.txt", "a") as f:
                f.write("\n".join(cleaned_lines))