
import csv

from runners.TFRunner import TFRunner;

model_type = "mlp"
model_complexity = "simple"

runner = TFRunner(model_type, model_complexity)

runner.define_model()
history = runner.train()

history_filename = f"{model_type}-{model_complexity}-history.csv"

with open(history_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    
    # Headers
    writer.writerow(['epoch'] + list(history.history.keys()))
    
    # Data
    for i in range(len(history.history['loss'])):
        writer.writerow([i + 1] + [history.history[key][i] for key in history.history])