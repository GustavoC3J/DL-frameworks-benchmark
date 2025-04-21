


import random
from abc import ABC, abstractmethod

import numpy as np


class Runner(ABC):

    def __init__(self, model_type, model_complexity, keras, epochs, batch_size, seed, gpu_ids, precision):

        self.model_type = model_type
        self.keras = keras
        self.model_complexity = model_complexity
        self.epochs = epochs
        self.batch_size = batch_size
        self.gpu_ids = [int(gpu) for gpu in gpu_ids.split(",") if gpu.isdigit()]
        self.precision = precision
        
        # Fix seed
        self.seed = seed
        random.seed(seed) # Python
        np.random.seed(seed) # NumPy/Pandas

    @abstractmethod
    def define_model(self):
        pass

    @abstractmethod
    def train(self, trainX, validX, trainY, validY):
        pass

    @abstractmethod
    def evaluate(self, testX, testY):
        pass