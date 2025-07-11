
import copy
import os
import time

import keras
import numpy as np
import torch
from keras.api.callbacks import ModelCheckpoint

from datasets.loader.data_loader_factory import DataLoaderFactory
from runners.model_builder.keras_model_builder import KerasModelBuilder
from runners.model_builder.torch_model_builder import TorchModelBuilder
from runners.runner import Runner
from utils.gpu_metrics import record_sample
from utils.metrics_callback import MetricsCallback
from utils.precision import Precision, get_keras_precision


class TorchRunner(Runner):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # Dataloader
        self.dl_factory = DataLoaderFactory("torch")
        
        # GPU
        if len(self.gpu_ids) == 1:
            # It is always 0, independently of CUDA index
            self.device = torch.device(f"cuda:0")
        else:
            raise NotImplementedError()
        
        # Fix the seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Set global floating point precision
        self.__set_precision()

    
    def __set_precision(self):
        if (self.keras):
            keras.config.set_dtype_policy(get_keras_precision(self.precision))

        else:
            self.amp = False # AMP = Automatic Mixed Precision

            if self.precision == Precision.FP32:
                self.dtype = torch.float32

            elif self.precision == Precision.FP16:
                self.dtype = torch.float16

            elif self.precision == Precision.BF16:
                self.dtype = torch.bfloat16

            else:
                self.dtype = torch.float32
                self.amp = True
                self.scaler = torch.amp.GradScaler()

                if self.precision == Precision.MIXED_FP16:
                    self.amp_dtype = torch.float16

                elif self.precision == Precision.MIXED_BF16:
                    self.amp_dtype = torch.bfloat16
                    
                else:
                    raise ValueError("Unsupported precision: " + self.precision)

    
    def define_model(self):

        if (self.keras):
            self.model = KerasModelBuilder(self.model_type, self.model_complexity).build()        
        else:
            self.model, self.config = TorchModelBuilder(self.model_type, self.model_complexity).build()

        
        # Move the model to the GPU and set it's precision
        if (not self.keras):
            self.model.to(device=self.device, dtype=self.dtype)

        # If there are multiple GPUs, 
        #if len(self.gpus) > 1:
            #self.model = torch.nn.DistributedDataParallel(self.model, device_ids=self.gpus)



    def __keras_train(self, train_dl, val_dl, path):

        checkpoint_filepath = path + "/model.keras"
        callbacks = [
            MetricsCallback(self.gpu_ids),
            ModelCheckpoint(
                filepath=checkpoint_filepath,
                monitor="val_loss",
                mode="min",
                save_best_only=True
            )
        ]
        
        # Train the model
        history = self.model.fit(
            train_dl,
            validation_data = val_dl,
            epochs = self.epochs,
            callbacks=callbacks
        )

        # Load best model
        if os.path.exists(checkpoint_filepath):
            self.model = keras.models.load_model(checkpoint_filepath)
    
        return history.history, callbacks[0].samples_logs
    

    def __torch_train(self, train_dl, val_dl, path):
        
        num_batches = len(train_dl)
        metric_name = self.config["metric_name"]
        samples_logs = []
        best_model_weights = None
        best_val_loss = float('inf')

        history = {
            "loss": [],
            metric_name: [],
            "val_loss": [],
            f"val_{metric_name}": [],
            "epoch_time": []
        }

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
                # Send data to GPU and set dtype
                # If output has to be an integer, then batch_y dtype is not modified
                batch_x = batch_x.to(device=self.device, dtype=self.dtype)
                batch_y = batch_y.to(device=self.device, dtype=torch.int64 if batch_y.dtype == torch.int64 else self.dtype)

                self.config["optimizer"].zero_grad()

                if self.amp:
                    # Get outputs and loss using lower precision
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                        outputs = self.model(batch_x)

                        if self.model_type == "lstm":
                            outputs = outputs.squeeze(1)

                        loss = self.config["loss_fn"](outputs, batch_y)
                        metric = self.config["metric_fn"](outputs, batch_y)
                        
                    # Perform updates in higher precision
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.config["optimizer"])
                    self.scaler.update()
                        
                else:
                    # Get loss and perform updates using the same precision
                    outputs = self.model(batch_x)
                    if self.model_type == "lstm":
                        outputs = outputs.squeeze(1)

                    loss = self.config["loss_fn"](outputs, batch_y)
                    metric = self.config["metric_fn"](outputs, batch_y)

                    loss.backward()
                    self.config["optimizer"].step()

                
                train_losses.append(loss.item())
                train_metrics.append(metric)

                # Check if a sample should be obtained
                samples_per_epoch = 4
                batches_per_sample = max(1, num_batches // samples_per_epoch)
        
                if (i != 0) and (i % batches_per_sample == 0):
                    sample = record_sample(start_time, self.gpu_ids)
                    sample["epoch"] = epoch
                    samples_logs.append(sample)

            # Validation
            val_logs, val_samples_logs = self.__torch_evaluate(val_dl, start_time)

            val_loss = val_logs["loss"]
            val_metric = val_logs[metric_name]
            samples_logs.extend(val_samples_logs)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(self.model.state_dict())


            # Save metrics
            history["loss"].append(np.mean(np.array(train_losses)))
            history[metric_name].append(np.mean(np.array(train_metrics)))
            history["val_loss"].append(val_loss)
            history[f"val_{metric_name}"].append(val_metric)
            history["epoch_time"].append(time.time() - epoch_start_time)

            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {history['loss'][-1]:.4f} - Val Loss: {val_loss:.4f} - Val {metric_name}: {val_metric:.4f}")

        # Save and load the best model
        if (best_model_weights != None):
            self.model.load_state_dict(best_model_weights)

        torch.save(self.model.state_dict(), path + f'/{epoch:02d}_model.pt')
        
        return history, samples_logs


    def train(self, trainX, validX, trainY, validY, path):
        train_dl = self.dl_factory.fromNumpy( trainX, trainY, self.batch_size, shuffle=(self.model != "lstm") )
        val_dl = self.dl_factory.fromNumpy( validX, validY, self.batch_size, shuffle=(self.model != "lstm") )

        train_fn = self.__keras_train if self.keras else self.__torch_train

        return train_fn(train_dl, val_dl, path)


    def __keras_evaluate(self, test_dl):
        callback = MetricsCallback(self.gpu_ids)

        self.model.evaluate(test_dl, callbacks=[callback])

        return callback.test_logs, callback.samples_logs
    

    def __torch_evaluate(self, test_dl, training_start_time = None):

        losses = []
        metrics = []
        num_batches = len(test_dl)
        samples_logs = []

        # Set evaluation mode
        self.model.eval()

        start_time = time.time()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_dl):
                # Send data to GPU
                batch_x = batch_x.to(device=self.device, dtype=self.dtype)
                batch_y = batch_y.to(device=self.device, dtype=torch.long if batch_y.dtype == torch.long else self.dtype)

                if self.amp:
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                        test_outputs = self.model(batch_x)

                        if self.model_type == "lstm":
                            test_outputs = test_outputs.squeeze(1)

                        loss = self.config["loss_fn"](test_outputs, batch_y)
                        metric = self.config["metric_fn"](test_outputs, batch_y)

                else:
                    test_outputs = self.model(batch_x)

                    if self.model_type == "lstm":
                        test_outputs = test_outputs.squeeze(1)

                    loss = self.config["loss_fn"](test_outputs, batch_y)
                    metric = self.config["metric_fn"](test_outputs, batch_y)

                losses.append(loss.item())
                metrics.append(metric)

                # Check if a sample should be obtained
                samples_per_epoch = 4
                batches_per_sample = max(1, num_batches // samples_per_epoch)
        
                if (i != 0) and (i % batches_per_sample == 0):
                    sample = record_sample(
                        start_time if training_start_time is None else training_start_time,
                        self.gpu_ids
                    )
                    samples_logs.append(sample)

        # Calculate mean
        test_loss = np.mean(np.array(losses))
        test_metric = np.mean(np.array(metrics))
    
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



