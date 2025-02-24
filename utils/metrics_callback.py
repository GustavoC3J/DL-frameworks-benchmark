
import time

import numpy as np
from utils.gpu_metrics import get_gpu_metrics
from keras.callbacks import Callback


class MetricsCallback(Callback):

    def __init__(self, gpu_indices, samples_per_epoch=4):
        super().__init__()
        
        self.samples_logs = []

        # Get visible GPUs
        self.gpu_indices = gpu_indices

        self.samples_per_epoch = samples_per_epoch

        # Used to avoid calling test methods during validation
        self.in_training = False
        
    

    # Training

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        # Number of batches after which a sample is obtained
        self.steps_per_sample = max(1, self.params['steps'] // self.samples_per_epoch)

    
    def on_epoch_begin(self, epoch, logs=None):
        self.in_training = True
        self.epoch_start_time = time.time()
        self.current_epoch = epoch
        self.get_validation_sample = True # Get one validation sample
    

    def on_epoch_end(self, epoch, logs=None):
        self.in_training = False
        logs['epoch_time'] = time.time() - self.epoch_start_time
        

    def on_batch_end(self, batch, logs=None):
        if (batch != 0) and (batch % self.steps_per_sample == 0):
            self.__record_sample(batch)

    

    # Testing
    
    def on_test_begin(self, logs=None):
        if self.in_training:  # Skip when validating (only test)
            return
        
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        
        # Number of batches after which a sample is obtained
        self.steps_per_sample = max(1, self.params['steps'] // self.samples_per_epoch)

    def on_test_end(self, logs=None):
        if self.in_training:
            return
        logs['epoch_time'] = time.time() - self.epoch_start_time
        self.test_logs = logs.copy()

    def on_test_batch_end(self, batch, logs=None):
        
        if not self.in_training:
            # Test sample
            if (batch != 0) and (batch % self.steps_per_sample == 0):
                self.__record_sample(batch)

        elif self.get_validation_sample:
            # Val sample
            self.get_validation_sample = False
            self.__record_sample(batch)

        
        
    # Metrics functions    
    
    def __record_sample(self, batch):
        

        sample = {
            "timestamp": time.time() - self.start_time # Add current sample timestamp
        }

        if (self.in_training):
            sample["epoch"] = self.current_epoch

        # GPU metrics
        gpu_metrics = get_gpu_metrics(self.gpu_indices)
        
        for gpu in gpu_metrics:
            gpu_id = gpu['index']
            prefix = f'gpu_{gpu_id}_'

            sample[prefix + 'utilization'] = gpu['utilization']
            sample[prefix + 'memory_used'] = gpu['memory_used']
            sample[prefix + 'power'] = gpu['power']

        self.samples_logs.append(sample)


