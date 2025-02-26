
import time
from runners.model_builder.torch_model_builder import TorchModelBuilder
import torch
import numpy as np

from runners.model_builder.keras_model_builder import KerasModelBuilder
from runners.runner import Runner
from utils.gpu_metrics import record_sample
from utils.metrics_callback import MetricsCallback
from utils.torch_utils import to_float_tensor, to_long_tensor


class TorchRunner(Runner):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        
        # Fix the seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    
    def define_model(self):

        if (self.keras):
            self.model = KerasModelBuilder(self.model_type, self.model_complexity).build()        
        else:
            self.model, self.config = TorchModelBuilder(self.model_type, self.model_complexity).build()

        
        # Move the model to the first GPU
        self.device = torch.device(f"cuda:{self.gpus[0]}")
        self.model.to(self.device)

        # If there are multiple GPUs, 
        if len(self.gpus) > 1:
            self.model = torch.nn.DistributedDataParallel(self.model, device_ids=self.gpus)



    def __keras_train(self, trainX, validX, trainY, validY):
        callback = MetricsCallback(self.gpus)
        
        # Train the model
        history = self.model.fit(
            trainX,
            trainY,
            epochs = self.epochs,
            batch_size = len(self.gpus) * self.batch_size,
            validation_data = (validX, validY),
            callbacks=[callback]
        )
    
        return history.history, callback.samples_logs
    

    def __torch_train(self, trainX, validX, trainY, validY):
        
        trainX = to_float_tensor(trainX, self.device)
        trainY = to_long_tensor( trainY, self.device)
        validX = to_float_tensor(validX, self.device)
        validY = to_long_tensor( validY, self.device)

        if (self.model_type == "cnn"):
            # Switch to (batch_size, channels, height, width)
            trainX = torch.permute(trainX, (0, 3, 2, 1))
            validX = torch.permute(validX, (0, 3, 2, 1))

        num_batches = len(trainX) // self.batch_size
        
        metric_name = self.config["metric_name"]

        history = {
            "loss": [],
            metric_name: [],
            "val_loss": [],
            f"val_{metric_name}": [],
            "epoch_time": []
        }

        samples_logs = []

        # Training start time
        start_time = time.time()

        # Entrenamiento
        for epoch in range(self.epochs):
            # Training
            epoch_start_time = time.time()
            train_losses = []
            train_metrics = []

            self.model.train()
            
            for i in range(num_batches):
                batch_x = trainX[i * self.batch_size : (i + 1) * self.batch_size]
                batch_y = trainY[i * self.batch_size : (i + 1) * self.batch_size]
                
                self.config["optimizer"].zero_grad()
                outputs = self.model(batch_x)
                loss = self.config["loss_fn"](outputs, batch_y)
                loss.backward()
                self.config["optimizer"].step()
                
                train_losses.append(loss.item())
                train_metrics.append(self.config["metric_fn"](outputs, batch_y))

                # Check if a sample should be obtained
                samples_per_epoch = 4
                batches_per_sample = max(1, num_batches // samples_per_epoch)
        
                if (i != 0) and (i % batches_per_sample == 0):
                    sample = record_sample(start_time, self.gpus)
                    sample["epoch"] = epoch
                    samples_logs.append(sample)

            # Validation
            test_outputs = self.model(validX)
            val_loss = self.config["loss_fn"](test_outputs, validY).item()
            val_metric = self.config["metric_fn"](test_outputs, validY)

            sample = record_sample(start_time, self.gpus)
            sample["epoch"] = epoch
            samples_logs.append(sample)

            # Save metrics
            history["loss"].append(np.mean(np.array(train_losses)))
            history[metric_name].append(np.mean(np.array(train_metrics)))
            history["val_loss"].append(val_loss)
            history[f"val_{metric_name}"].append(val_metric)
            history["epoch_time"].append(time.time() - epoch_start_time)

            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {history['loss'][-1]:.4f} - Val Loss: {val_loss:.4f} - Val {metric_name}: {val_metric:.4f}")

        return history, samples_logs


    def train(self, trainX, validX, trainY, validY):
        train_fn = self.__keras_train if self.keras else self.__torch_train
        print(type(trainX), type(validX), type(trainY), type(validY))
        return train_fn(trainX, validX, trainY, validY)


    def __keras_evaluate(self, testX, testY):
        callback = MetricsCallback(self.gpus)

        self.model.evaluate(testX, testY, callbacks=[callback])

        return callback.test_logs, callback.samples_logs
    

    def __torch_evaluate(self, testX, testY):
        testX = to_float_tensor(testX, self.device)
        testY = to_long_tensor(testY, self.device)

        if (self.model_type == "cnn"):
            # Switch to (batch_size, channels, height, width)
            testX = torch.permute(testX, (0, 3, 2, 1))

        start = time.time()

        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(testX)
            test_loss = self.config["loss_fn"](test_outputs, testY).item()
            test_metric = self.config["metric_fn"](test_outputs, testY)

        samples_logs = [record_sample(start, self.gpus)]
    
        print(f"Loss: {test_loss:.4f} - {self.config['metric_name']}: {test_metric:.4f}")

        test_logs = {
            "loss": test_loss,
            self.config['metric_name']: test_metric,
            "epoch_time": time.time() - start
        }

        return test_logs, samples_logs


    
    def evaluate(self, testX, testY):
        evaluate_fn = self.__keras_evaluate if self.keras else self.__torch_evaluate
        return evaluate_fn(testX, testY)



