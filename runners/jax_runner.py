
import jax
import jax.numpy as jnp
from jax import random

from runners.model_builder.keras_model_builder import KerasModelBuilder
from runners.runner import Runner
from utils.metrics_callback import MetricsCallback


class JaxRunner(Runner):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        
        # Fix the seed
        self.key = random.PRNGKey(self.seed)
        
        self.devices = jax.devices()

    
    def define_model(self):

        if len(self.gpus) > 1:
            raise NotImplementedError("Multiple GPU training is not implemented")
        
        jax.config.update("jax_default_device", jax.devices("gpu")[self.gpus[0]])

        self.model = KerasModelBuilder(self.model_type, self.model_complexity).build()


    def train(self, trainX, validX, trainY, validY):
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


    def evaluate(self, testX, testY):
        callback = MetricsCallback(self.gpus)

        self.model.evaluate(testX, testY, callbacks=[callback])

        return callback.test_logs, callback.samples_logs



