
import time
import jax
import jax.numpy as jnp

from runners.model_builder.keras_model_builder import KerasModelBuilder
from runners.model_builder.flax_model_builder import FlaxModelBuilder
from runners.runner import Runner
from utils.gpu_metrics import get_gpu_metrics
from utils.metrics_callback import MetricsCallback
from utils.jax_utils import TrainState, classif_train_step, classif_eval_step, regression_eval_step, regression_train_step


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
                tx=self.config["optimizer"],
                batch_stats=self.config.get("batch_stats", None)
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


    def __record_sample(self):

        sample = {
            "timestamp": time.time() - self.start_time # Add current sample timestamp
        }

        # GPU metrics
        gpu_metrics = get_gpu_metrics(self.gpus)
        
        for gpu in gpu_metrics:
            gpu_id = gpu['index']
            prefix = f'gpu_{gpu_id}_'

            sample[prefix + 'utilization'] = gpu['utilization']
            sample[prefix + 'memory_used'] = gpu['memory_used']
            sample[prefix + 'power'] = gpu['power']

        return sample
    




    def __jax_train(self, trainX, validX, trainY, validY):
        # Training loop using JAX

        # Parse data into JAX arrays
        trainX, trainY = jnp.array(trainX), jnp.array(trainY)
        validX, validY = jnp.array(validX), jnp.array(validY)
        
        metric_name = self.config["metric_name"]

        history = {
            "loss": [],
            metric_name: [],
            "val_loss": [],
            f"val_{metric_name}": [],
            "epoch_time": []
        }

        samples_logs = []

        num_batches = len(trainX) // self.batch_size

        train_step_fn = regression_train_step if self.model_type == "lstm" else classif_train_step
        eval_step_fn = regression_eval_step if self.model_type == "lstm" else classif_eval_step

        # Training start time
        self.start_time = time.time()

        for epoch in range(self.epochs):
            # Training
            epoch_start_time = time.time()
            train_losses = []
            train_metrics = []
            
            for i in range(num_batches):
                batch_x = trainX[i * self.batch_size : (i + 1) * self.batch_size]
                batch_y = trainY[i * self.batch_size : (i + 1) * self.batch_size]

                self.key, subkey = jax.random.split(self.key)
                self.state, loss, metric = train_step_fn(self.state, (batch_x, batch_y), subkey)
                train_losses.append(loss)
                train_metrics.append(metric)

                # Check if a sample should be obtained
                samples_per_epoch = 4
                batches_per_sample = max(1, num_batches // samples_per_epoch)
        
                if (i != 0) and (i % batches_per_sample == 0):
                    sample = self.__record_sample()
                    sample["epoch"] = epoch
                    samples_logs.append(sample)

            # Validation
            val_loss, val_accuracy = eval_step_fn(self.state, (jnp.array(validX), jnp.array(validY)))
            sample = self.__record_sample()
            sample["epoch"] = epoch
            samples_logs.append(sample)

            # Save metrics
            history["loss"].append(jnp.mean(jnp.array(train_losses)).item())
            history[metric_name].append(jnp.mean(jnp.array(train_metrics)).item())
            history["val_loss"].append(val_loss.item())
            history[f"val_{metric_name}"].append(val_accuracy.item())
            history["epoch_time"].append(time.time() - epoch_start_time)

            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {history['loss'][-1]:.4f} - Val Loss: {val_loss:.4f} - Val {metric_name}: {val_accuracy:.4f}")
        
        return history, samples_logs
    
    def train(self, trainX, validX, trainY, validY):
        return self.__keras_train(trainX, validX, trainY, validY) if self.keras else self.__jax_train(trainX, validX, trainY, validY)


    def __keras_evaluate(self, testX, testY):
        callback = MetricsCallback(self.gpus)

        self.model.evaluate(testX, testY, callbacks=[callback])

        return callback.test_logs, callback.samples_logs
    

    def __jax_evaluate(self, testX, testY):

        eval_step_fn = regression_eval_step if self.model_type == "lstm" else classif_eval_step
        
        start = time.time()
        test_loss, test_metric = eval_step_fn( self.state, (jnp.array(testX), jnp.array(testY)) )
        samples_logs = [self.__record_sample()]

        print(f"Loss: {test_loss:.4f} - {self.config['metric_name']}: {test_metric:.4f}")

        test_logs = {
            "loss": test_loss,
            self.config['metric_name']: test_metric,
            "epoch_time": time.time() - start
        }

        return test_logs, samples_logs

    
    def evaluate(self, testX, testY):
        return self.__keras_evaluate(testX, testY) if self.keras else self.__jax_evaluate(testX, testY)
        



