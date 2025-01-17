
import csv
import os
from datetime import datetime

from runners.TFRunner import TFRunner;

# Parameters
model_type = "mlp"
model_complexity = "complex"
epochs = 3


# Create output directory
output_directory = "results/"
os.makedirs(output_directory, exist_ok=True)

# Generate timestamp for results filename
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
results_filename = f"{timestamp}-{model_type}-{model_complexity}.csv"


# Path to the results file
results_filepath = os.path.join(output_directory, results_filename)



# Perform the experiment
runner = TFRunner(model_type, model_complexity, epochs)

runner.define_model()
history = runner.train()


# Save the results
with open(results_filepath, mode='w', newline='') as f:
    writer = csv.writer(f)
    
    # Headers
    writer.writerow(['epoch'] + list(history.history.keys()))
    
    # Data
    for i in range(len(history.history['loss'])):
        writer.writerow([i + 1] + [history.history[key][i] for key in history.history])