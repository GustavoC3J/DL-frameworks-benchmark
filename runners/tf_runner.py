
import keras
import tensorflow as tf
from keras.api.callbacks import ModelCheckpoint

from datasets.data_loader_factory import DataLoaderFactory
from runners.model_builder.keras_model_builder import KerasModelBuilder
from runners.runner import Runner
from utils.metrics_callback import MetricsCallback
from utils.precision import Precision, get_keras_precision


class TFRunner(Runner):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # Dataloader
        self.dl_factory = DataLoaderFactory("tf")
        
        # Fix the seed
        tf.random.set_seed(self.seed)

        # Set global floating point precision
        if self.model_type == "lstm" and self.precision == Precision.BF16:
            precision = "float16"
        else:
            precision = get_keras_precision(self.precision)

        keras.config.set_dtype_policy(precision)

    
    def define_model(self):
    
        # Define the strategy to follow in order to balance the workload between GPUs
        if len(self.gpu_ids) > 1:
            strategy = tf.distribute.MirroredStrategy( [f"GPU:{gpu}" for gpu in self.gpu_ids] )
        else:
            strategy = tf.distribute.get_strategy()
        
        with strategy.scope():
            self.model = KerasModelBuilder(self.model_type, self.model_complexity).build()



    def train(self, trainX, validX, trainY, validY, path):
        train_dl = self.dl_factory.fromNumpy( trainX, trainY, self.batch_size, shuffle=(self.model != "lstm") )
        val_dl = self.dl_factory.fromNumpy( validX, validY, self.batch_size, shuffle=(self.model != "lstm") )

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
        keras.models.load_model(checkpoint_filepath)
    
        return history.history, callbacks[0].samples_logs


    def evaluate(self, testX, testY):
        test_dl = self.dl_factory.fromNumpy(testX, testY, self.batch_size, shuffle=False)
        
        callback = MetricsCallback(self.gpu_ids)

        self.model.evaluate(test_dl, callbacks=[callback])

        return callback.test_logs, callback.samples_logs



