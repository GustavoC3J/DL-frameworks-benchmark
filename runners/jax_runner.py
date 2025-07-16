
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jmp
import keras
import orbax.checkpoint
from flax.training import checkpoints
from keras.api.callbacks import ModelCheckpoint

from datasets.loader.data_loader_factory import DataLoaderFactory
from runners.model_builder.flax_model_builder import FlaxModelBuilder
from runners.model_builder.keras_model_builder import KerasModelBuilder
from runners.runner import Runner
from utils.gpu_metrics import record_sample
from utils.jax_utils import TrainState, make_eval_step, make_train_step
from utils.metrics_callback import MetricsCallback
from utils.precision import Precision, get_keras_precision


class JaxRunner(Runner):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        
        # Dataloader
        self.dl_factory = DataLoaderFactory("torch")
        
        # Fix the seed
        self.key = jax.random.key(seed=self.seed)

        # Set GPUs
        if len(self.gpu_ids) > 1:
            raise NotImplementedError("Multiple GPU training is not implemented")
        elif len(self.gpu_ids) == 1:
            jax.config.update("jax_default_device", jax.devices("gpu")[0])

        # Set global floating point precision
        self.__set_precision()
        
    
    def __set_precision(self):
        if (self.keras):
            keras.config.set_dtype_policy(get_keras_precision(self.precision))

        else:
            self.loss_scale = None

            if self.precision == Precision.FP32:
                self.policy = jmp.get_policy("float32")
            
            elif self.precision == Precision.FP16:
                self.policy = jmp.get_policy("float16")

            elif self.precision == Precision.BF16:
                self.policy = jmp.get_policy("bfloat16")

            else:
                self.loss_scale = jmp.DynamicLossScale(jnp.float32(2 ** 15))

                if self.precision == Precision.MIXED_FP16:
                    self.policy = jmp.Policy(
                        compute_dtype=jnp.float16,
                        param_dtype=jnp.float32,
                        output_dtype=jnp.float32
                    )
                elif self.precision == Precision.MIXED_BF16:
                    self.policy = jmp.Policy(
                        compute_dtype=jnp.bfloat16,
                        param_dtype=jnp.float32,
                        output_dtype=jnp.float32
                    )
                else:
                    raise ValueError("Unsupported precision: " + self.precision)

    
    def define_model(self):

        if self.keras:
            self.model = KerasModelBuilder(self.model_type, self.model_complexity).build()
        else:
            self.key, subkey = jax.random.split(self.key)
            self.model, self.config = FlaxModelBuilder(self.model_type, self.model_complexity, subkey, self.policy).build()

            # Set up the state using the model and cofiguration
            self.state = TrainState.create(
                apply_fn=self.model.apply,
                params=self.config["params"],
                tx=self.config["optimizer"],
                batch_stats=self.config.get("batch_stats", None),
                loss_scale=self.loss_scale
            )



    def __keras_train(self, train_dl, val_dl, path):
        # Training using Keras

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


    def __jax_train(self, train_dl, val_dl, path):
        # Training loop using JAX
        
        metric_name = self.config["metric_name"]
        best_model_weights = None
        best_val_loss = float('inf')

        history = {
            "loss": [],
            metric_name: [],
            "val_loss": [],
            f"val_{metric_name}": [],
            "epoch_time": []
        }

        samples_logs = []

        num_batches = len(train_dl)

        train_step = make_train_step(self.config["loss_fn"], self.config["metric_fn"])

        # Training start time
        start_time = time.time()

        for epoch in range(self.epochs):
            # Training
            epoch_start_time = time.time()
            train_losses = []
            train_metrics = []
            
            for i, (batch_x, batch_y) in enumerate(train_dl):
                # Parse data into JAX arrays
                batch_x = jnp.array(batch_x)
                batch_y = jnp.array(batch_y)

                self.key, subkey = jax.random.split(self.key)
                self.state, loss, metric = train_step(self.state, (batch_x, batch_y), subkey)
                train_losses.append(loss)
                train_metrics.append(metric)

                # Check if a sample should be obtained
                samples_per_epoch = 4
                batches_per_sample = max(1, num_batches // samples_per_epoch)
        
                if (i != 0) and (i % batches_per_sample == 0):
                    sample = record_sample(start_time, self.gpu_ids)
                    sample["epoch"] = epoch
                    samples_logs.append(sample)

            # Validation
            val_logs, val_samples_logs = self.__jax_evaluate(val_dl, start_time)

            val_loss = val_logs["loss"]
            val_metric = val_logs[metric_name]
            samples_logs.extend(val_samples_logs)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = self.state.replace()

            # Save metrics
            history["loss"].append(jnp.mean(jnp.array(train_losses)).item())
            history[metric_name].append(jnp.mean(jnp.array(train_metrics)).item())
            history["val_loss"].append(val_loss)
            history[f"val_{metric_name}"].append(val_metric)
            history["epoch_time"].append(time.time() - epoch_start_time)

            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {history['loss'][-1]:.4f} - Val Loss: {val_loss:.4f} - Val {metric_name}: {val_metric:.4f}")

        # Save and load the best model
        if (best_model_weights != None):
            self.state = best_model_weights

        checkpoints.save_checkpoint(
            Path(path).absolute(),
            self.state.replace(loss_scale=None), # Not needed anymore, and it's incompatible with checkpointing,
            0,
            orbax_checkpointer=orbax.checkpoint.PyTreeCheckpointer()
        )
        
        return history, samples_logs
    
    def train(self, trainX, validX, trainY, validY, path):
        train_dl = self.dl_factory.fromNumpy( trainX, trainY, self.batch_size, shuffle=(self.model != "lstm") )
        val_dl = self.dl_factory.fromNumpy( validX, validY, self.batch_size, shuffle=(self.model != "lstm") )

        return self.__keras_train(train_dl, val_dl, path) if self.keras else self.__jax_train(train_dl, val_dl, path)


    def __keras_evaluate(self, test_dl):
        callback = MetricsCallback(self.gpu_ids)

        self.model.evaluate(test_dl, callbacks=[callback])

        return callback.test_logs, callback.samples_logs
    

    def __jax_evaluate(self, test_dl, training_start_time = None):

        test_loss = 0
        test_metric = 0
        num_batches = len(test_dl)
        samples_logs = []
        
        eval_step = make_eval_step(self.config["loss_fn"], self.config["metric_fn"])
        
        start_time = time.time()
        for i, (batch_x, batch_y) in enumerate(test_dl):
            loss, metric = eval_step(self.state, (jnp.array(batch_x), jnp.array(batch_y)))

            test_loss += loss
            test_metric += metric

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
        test_loss /= num_batches
        test_metric /= num_batches
        
        test_logs = {
            "loss": test_loss.item(),
            self.config['metric_name']: test_metric.item(),
            "epoch_time": time.time() - start_time
        }
        
        # Print log message if it is test
        if training_start_time is None:
            print(f"Loss: {test_loss.item():.4f} - {self.config['metric_name']}: {test_metric.item():.4f}")

        return test_logs, samples_logs

    
    def evaluate(self, testX, testY):
        test_dl = self.dl_factory.fromNumpy(testX, testY, self.batch_size, shuffle=False)

        return self.__keras_evaluate(test_dl) if self.keras else self.__jax_evaluate(test_dl)
        



