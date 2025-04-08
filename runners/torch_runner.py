
import time
from runners.model_builder.torch_model_builder import TorchModelBuilder
import torch
import numpy as np

from runners.model_builder.keras_model_builder import KerasModelBuilder
from runners.runner import Runner
from utils.gpu_metrics import record_sample
from utils.metrics_callback import MetricsCallback
from datasets.data_loader_factory import DataLoaderFactory


class TorchRunner(Runner):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # Dataloader
        self.dl_factory = DataLoaderFactory("torch")
        
        # GPU
        self.device = torch.device(f"cuda:{self.gpus[0]}")
        
        # Fix the seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    
    def define_model(self):

        if (self.keras):
            self.model = KerasModelBuilder(self.model_type, self.model_complexity).build()        
        else:
            self.model, self.config = TorchModelBuilder(self.model_type, self.model_complexity).build()

        
        # Move the model to the GPU
        self.model.to(self.device)

        # If there are multiple GPUs, 
        if len(self.gpus) > 1:
            self.model = torch.nn.DistributedDataParallel(self.model, device_ids=self.gpus)



    def __keras_train(self, train_dl, val_dl):
        callback = MetricsCallback(self.gpus)
        
        # Train the model
        history = self.model.fit(
            train_dl,
            validation_data = val_dl,
            epochs = self.epochs,
            callbacks=[callback]
        )
    
        return history.history, callback.samples_logs
    

    def __torch_train(self, train_dl, val_dl):
        
        num_batches = len(train_dl)
        
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

            # Set training mode
            self.model.train()
            
            for i, (batch_x, batch_y) in enumerate(train_dl):

                # Send data to GPU
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.config["optimizer"].zero_grad()

                outputs = self.model(batch_x)
                if self.model_type == "lstm": outputs = outputs.squeeze(1)

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
            val_logs, val_samples_logs = self.__torch_evaluate(val_dl, start_time)

            val_loss = val_logs["loss"]
            val_metric = val_logs[metric_name]
            samples_logs.extend(val_samples_logs)


            # Save metrics
            history["loss"].append(np.mean(np.array(train_losses)))
            history[metric_name].append(np.mean(np.array(train_metrics)))
            history["val_loss"].append(val_loss)
            history[f"val_{metric_name}"].append(val_metric)
            history["epoch_time"].append(time.time() - epoch_start_time)

            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {history['loss'][-1]:.4f} - Val Loss: {val_loss:.4f} - Val {metric_name}: {val_metric:.4f}")

        return history, samples_logs


    def train(self, trainX, validX, trainY, validY):
        train_dl = self.dl_factory.fromNumpy( trainX, trainY, self.batch_size, shuffle=(self.model != "lstm") )
        val_dl = self.dl_factory.fromNumpy( validX, validY, self.batch_size, shuffle=(self.model != "lstm") )

        train_fn = self.__keras_train if self.keras else self.__torch_train

        return train_fn(train_dl, val_dl)


    def __keras_evaluate(self, test_dl):
        callback = MetricsCallback(self.gpus)

        self.model.evaluate(test_dl, callbacks=[callback])

        return callback.test_logs, callback.samples_logs
    

    def __torch_evaluate(self, test_dl, training_start_time = None):

        test_loss = 0
        test_metric = 0
        num_batches = len(test_dl)
        samples_logs = []

        # Set evaluation mode
        self.model.eval()

        start_time = time.time()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_dl):
                # Send data to GPU
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                test_outputs = self.model(batch_x)
                if self.model_type == "lstm":
                    test_outputs = test_outputs.squeeze(1)

                test_loss += self.config["loss_fn"](test_outputs, batch_y).item()
                test_metric += self.config["metric_fn"](test_outputs, batch_y)

                # Check if a sample should be obtained
                samples_per_epoch = 4
                batches_per_sample = max(1, num_batches // samples_per_epoch)
        
                if (i != 0) and (i % batches_per_sample == 0):
                    sample = record_sample(
                        start_time if training_start_time is None else training_start_time,
                        self.gpus
                    )
                    samples_logs.append(sample)

        # Calculate mean
        test_loss /= num_batches
        test_metric /= num_batches
    
        # Print log message if it is test
        if training_start_time is None:
            print(f"Loss: {test_loss:.4f} - {self.config['metric_name']}: {test_metric:.4f}")

        test_logs = {
            "loss": test_loss,
            self.config['metric_name']: test_metric,
            "epoch_time": time.time() - start_time
        }

        return test_logs, samples_logs


    
    def evaluate(self, testX, testY):
        test_dl = self.dl_factory.fromNumpy(testX, testY, self.batch_size, shuffle=False)

        evaluate_fn = self.__keras_evaluate if self.keras else self.__torch_evaluate

        return evaluate_fn(test_dl)



