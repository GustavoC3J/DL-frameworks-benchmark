

import time
import GPUtil
import numpy as np
from tensorflow.keras.callbacks import Callback


class TFMetricsCallback(Callback):

    def __init__(self):
        super().__init__()
        self.gpus = len(GPUtil.getGPUs())
        self.samples_per_epoch = 4
        self.sample_index = 0
        self.gpu_batch_metrics = [{} for _ in range(self.gpus)]
    
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

        # Reset the usage metrics
        self.sample_index = 0

        for i in range(self.gpus):
            self.gpu_batch_metrics[i] = {
                'load': np.zeros(self.samples_per_epoch),
                'memory_used': np.zeros(self.samples_per_epoch)
            } 

    def on_epoch_end(self, epoch, logs=None):
        # Epoch time 
        elapsed_time = time.time() - self.start_time
        logs['epoch_time'] = elapsed_time  

        # GPUs metrics
        for i, gpu in enumerate(GPUtil.getGPUs()):
            prefix = f'gpu_{i}_'
            logs[prefix + 'load'] = np.mean(self.gpu_batch_metrics[i]['load'])
            logs[prefix + 'memory_used'] = np.mean(self.gpu_batch_metrics[i]['memory_used'])
            logs[prefix + 'memory_total'] = gpu.memoryTotal



    def on_batch_end(self, batch, logs=None):
        # Number of batches after which a sample is obtained
        steps_per_sample = self.params['steps'] // self.samples_per_epoch
        
        if (batch % steps_per_sample == 0):

            # GPU metrics
            for i, gpu in enumerate(GPUtil.getGPUs()):
                self.gpu_batch_metrics[i]['load'][self.sample_index] = gpu.load
                self.gpu_batch_metrics[i]['memory_used'][self.sample_index] = gpu.memoryUsed
            self.sample_index += 1
        

            
            