import os
import time

import numpy as np
from utils.gpu_metrics import get_gpu_metrics
from tensorflow.keras.callbacks import Callback


class TFMetricsCallback(Callback):

    def __init__(self, samples_per_epoch=4):
        super().__init__()
        
        # Get visible GPUs
        visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self.gpu_indices = [int(gpu) for gpu in visible_gpus.split(",") if gpu.isdigit()]
        
        self.samples_per_epoch = samples_per_epoch
        self.in_training = False # Used to avoid calling test methods during validation
        
    
    # Training
    
    def on_epoch_begin(self, epoch, logs=None):
        self.in_training = True
        self.__reset_metrics()
    

    def on_epoch_end(self, epoch, logs=None):
        self.in_training = False
        self.__compute_final_metrics(logs)
        

    def on_batch_end(self, batch, logs=None):
        self.__record_batch_metrics(batch)

    
    # Testing
    
    def on_test_begin(self, logs=None):
        if self.in_training:  # Skip when validating (only test)
            return
        self.__reset_metrics()
        self.test_logs = {}

    def on_test_end(self, logs=None):
        if self.in_training:
            return
        self.__compute_final_metrics(logs)
        self.test_logs = logs.copy()

    def on_test_batch_end(self, batch, logs=None):
        if self.in_training:
            return
        self.__record_batch_metrics(batch)
        
        
    # Metrics functions

    def __reset_metrics(self):
        # Reset usage metrics
        self.sample_index = 0
        self.gpu_batch_metrics = self.__init_gpu_metrics()
        self.start_time = time.time()

    def __init_gpu_metrics(self):
        # Initialize GPU metrics
        return {
            gpu: {
                'utilization': np.zeros(self.samples_per_epoch),
                'memory_used': np.zeros(self.samples_per_epoch),
                'power': np.zeros(self.samples_per_epoch)
            } for gpu in self.gpu_indices
        }
    
    
    def __record_batch_metrics(self, batch):
        # Number of batches after which a sample is obtained
        steps_per_sample = self.params['steps'] // self.samples_per_epoch
        
        if (batch != 0) and (batch % steps_per_sample == 0):
            gpu_metrics = get_gpu_metrics(self.gpu_indices)
            
            # GPU metrics
            for gpu in gpu_metrics:
                gpu_id = gpu['index']
                self.gpu_batch_metrics[gpu_id]['utilization'][self.sample_index] = gpu['utilization']
                self.gpu_batch_metrics[gpu_id]['memory_used'][self.sample_index] = gpu['memory_used']
                self.gpu_batch_metrics[gpu_id]['power'][self.sample_index] = gpu['power']
            


    def __compute_final_metrics(self, logs):
        # Epoch time 
        elapsed_time = time.time() - self.start_time
        logs['epoch_time'] = elapsed_time  
        
        # GPUs metrics
        gpu_metrics = get_gpu_metrics(self.gpu_indices)
        
        for gpu in gpu_metrics:
            gpu_id = gpu['index']
            prefix = f'gpu_{gpu_id}_'
            logs[prefix + 'utilization'] = np.mean(self.gpu_batch_metrics[gpu_id]['utilization'])
            logs[prefix + 'memory_used'] = np.mean(self.gpu_batch_metrics[gpu_id]['memory_used'])
            logs[prefix + 'power'] = np.mean(self.gpu_batch_metrics[gpu_id]['power'])