
import tensorflow as tf

from runners.model_builder.keras_model_builder import KerasModelBuilder
from runners.runner import Runner
from utils.tf_metrics_callback import TFMetricsCallback


class TFRunner(Runner):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        
        # Fix the seed
        tf.random.set_seed(self.seed)

    
    def define_model(self):
    
        # Define the strategy to follow in order to balance the workload between GPUs
        if len(self.gpus) > 1:
            strategy = tf.distribute.MirroredStrategy( [f"GPU:{gpu}" for gpu in self.gpus] )
        else:
            strategy = tf.distribute.get_strategy()
        
        with strategy.scope():
            self.model = KerasModelBuilder(self.model_type, self.model_complexity).build()



    def train(self, trainX, validX, trainY, validY):
        
        # Train the model
        return self.model.fit(
            trainX,
            trainY,
            epochs = self.epochs,
            batch_size = len(self.gpus) * self.batch_size,
            validation_data = (validX, validY),
            callbacks=[TFMetricsCallback(self.gpus)]
        )


    def evaluate(self, testX, testY):
        callback = TFMetricsCallback(self.gpus)

        self.model.evaluate(testX, testY, callbacks=[callback])

        return callback.test_logs



