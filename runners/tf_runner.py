
from datasets.data_loader_factory import DataLoaderFactory
import tensorflow as tf
import keras

from runners.model_builder.keras_model_builder import KerasModelBuilder
from runners.runner import Runner
from utils.metrics_callback import MetricsCallback


class TFRunner(Runner):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # Dataloader
        self.dl_factory = DataLoaderFactory("tf")
        
        # Fix the seed
        tf.random.set_seed(self.seed)

        # Set global floating point precision
        keras.config.set_dtype_policy(self.precision)

    
    def define_model(self):
    
        # Define the strategy to follow in order to balance the workload between GPUs
        if len(self.gpus) > 1:
            strategy = tf.distribute.MirroredStrategy( [f"GPU:{gpu}" for gpu in self.gpus] )
        else:
            strategy = tf.distribute.get_strategy()
        
        with strategy.scope():
            self.model = KerasModelBuilder(self.model_type, self.model_complexity).build()



    def train(self, trainX, validX, trainY, validY):
        train_dl = self.dl_factory.fromNumpy( trainX, trainY, self.batch_size, shuffle=(self.model != "lstm") )
        val_dl = self.dl_factory.fromNumpy( validX, validY, self.batch_size, shuffle=(self.model != "lstm") )

        callback = MetricsCallback(self.gpus)
        
        # Train the model
        history = self.model.fit(
            train_dl,
            validation_data = val_dl,
            epochs = self.epochs,
            callbacks=[callback]
        )
    
        return history.history, callback.samples_logs


    def evaluate(self, testX, testY):
        test_dl = self.dl_factory.fromNumpy(testX, testY, self.batch_size, shuffle=False)
        
        callback = MetricsCallback(self.gpus)

        self.model.evaluate(test_dl, callbacks=[callback])

        return callback.test_logs, callback.samples_logs



