
import time
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import numpy as np

from runners.model_builder.keras_model_builder import KerasModelBuilder
from runners.model_builder.flax_model_builder import FlaxModelBuilder
from runners.runner import Runner
from utils.metrics_callback import MetricsCallback
from utils.jax_utils import mlp_train_step, mlp_eval_step


class JaxRunner(Runner):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        
        # Fix the seed
        self.key = jax.random.key(seed=self.seed)
        
        self.devices = jax.devices()

    
    def define_model(self):

        if len(self.gpus) > 1:
            raise NotImplementedError("Multiple GPU training is not implemented")
        
        jax.config.update("jax_default_device", jax.devices("gpu")[self.gpus[0]])
        

        if self.keras:
            self.model = KerasModelBuilder(self.model_type, self.model_complexity).build()
        else:
            self.key, subkey = jax.random.split(self.key)
            self.model, self.config = FlaxModelBuilder(self.model_type, self.model_complexity, subkey).build()

            # Set up the state using the model and cofiguration
            self.state = TrainState.create(
                apply_fn=self.model.apply,
                params=self.config["params"],
                tx=self.config["optimizer"]
            )



    def __keras_train(self, trainX, validX, trainY, validY):
        # Training using Keras

        callback = MetricsCallback(self.gpus)
        
        history = self.model.fit(
            trainX,
            trainY,
            epochs = self.epochs,
            batch_size = len(self.gpus) * self.batch_size,
            validation_data = (validX, validY),
            callbacks=[callback]
        )
    
        return history.history, callback.samples_logs


    def __jax_train(self, trainX, validX, trainY, validY):
        # Training loop using JAX

        # Parse data into JAX arrays
        trainX, trainY = jnp.array(trainX), jnp.array(trainY)
        validX, validY = jnp.array(validX), jnp.array(validY)

        history = {
            "epoch": [],
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "epoch_time": []
        }

        num_batches = len(trainX) // self.batch_size

        for epoch in range(self.epochs):
            # Training
            start = time.time()
            train_losses = []
            train_metrics = []
            
            for i in range(num_batches):
                batch_x = trainX[i * self.batch_size : (i + 1) * self.batch_size]
                batch_y = trainY[i * self.batch_size : (i + 1) * self.batch_size]

                self.key, subkey = jax.random.split(self.key)
                self.state, loss, metric = mlp_train_step(self.state, (batch_x, batch_y), subkey)
                train_losses.append(loss)
                train_metrics.append(metric)

            # Validation
            val_loss, val_accuracy = mlp_eval_step(self.state, (jnp.array(validX), jnp.array(validY)))

            # Save metrics
            metric_name = self.config["metric_name"]

            history["loss"].append(jnp.mean(jnp.array(train_losses)).item())
            history[metric_name].append(jnp.mean(jnp.array(train_metrics)).item())
            history["val_loss"].append(val_loss.item())
            history[f"val_{metric_name}"].append(val_accuracy.item())
            history["epoch_time"].append(time.time() - start)

            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {history['loss'][-1]:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f}")

        return history, []
    
    def train(self, trainX, validX, trainY, validY):
        return self.__keras_train(trainX, validX, trainY, validY) if self.keras else self.__jax_train(trainX, validX, trainY, validY)


    def __keras_evaluate(self, testX, testY):
        callback = MetricsCallback(self.gpus)

        self.model.evaluate(testX, testY, callbacks=[callback])

        return callback.test_logs, callback.samples_logs
    

    def __jax_evaluate(self, testX, testY):
        
        start = time.time()
        test_loss, test_accuracy = mlp_eval_step( self.state, (jnp.array(testX), jnp.array(testY)) )

        print(f"Loss: {test_loss:.4f} - Accuracy: {test_accuracy:.4f}")

        test_logs = {
            "loss": test_loss,
            "accuracy": test_accuracy,
            "epoch_time": time.time() - start
        }

        samples_logs = {}

        return test_logs, samples_logs

    
    def evaluate(self, testX, testY):
        return self.__keras_evaluate(testX, testY) if self.keras else self.__jax_evaluate(testX, testY)
        



