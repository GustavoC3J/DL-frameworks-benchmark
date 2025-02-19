
import torch
import keras

from runners.model_builder.keras_model_builder import KerasModelBuilder
from runners.runner import Runner
from utils.metrics_callback import MetricsCallback


class TorchRunner(Runner):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        
        # Fix the seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    
    def define_model(self):

        self.model = KerasModelBuilder(self.model_type, self.model_complexity).build()

        # Move the model to the first GPU
        self.model.to(f"cuda:{self.gpus[0]}")

        # If there are multiple GPUs, 
        if len(self.gpus) > 1:
            self.model = torch.nn.DistributedDataParallel(self.model, device_ids=self.gpus)



    def train(self, trainX, validX, trainY, validY):
        
        # Train the model
        return self.model.fit(
            trainX,
            trainY,
            epochs = self.epochs,
            batch_size = len(self.gpus) * self.batch_size,
            validation_data = (validX, validY),
            callbacks=[MetricsCallback(self.gpus)]
        )


    def evaluate(self, testX, testY):
        callback = MetricsCallback(self.gpus)

        self.model.evaluate(testX, testY, callbacks=[callback])

        return callback.test_logs



