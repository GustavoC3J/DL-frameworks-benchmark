

import os
import pandas as pd
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

pd.DataFrame(history.history).to_csv(results_filepath)
