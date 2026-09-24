import random
from abc import ABC, abstractmethod

import numpy as np

from datasets.loader.data_loader_factory import DataLoaderFactory


class Runner(ABC):

    # Data pipeline of the backend: tf.data in TensorFlow, torch's DataLoader everywhere else
    data_framework = None

    # Batches each graph is warmed up with before the measured training starts
    PRECOMPILE_STEPS = 2

    def __init__(self, model_type, model_complexity, epochs, batch_size, seed, gpu_ids, precision):

        self.model_type = model_type
        self.model_complexity = model_complexity
        self.epochs = epochs
        self.batch_size = batch_size
        self.gpu_ids = [int(gpu) for gpu in gpu_ids.split(",") if gpu.isdigit()]
        self.precision = precision

        # Multi-GPU is not supported
        if len(self.gpu_ids) > 1:
            raise NotImplementedError("Only a single GPU is supported")

        self.dl_factory = DataLoaderFactory(self.data_framework)
        self.__loaders = None
        
        # Fix seed
        self.seed = seed
        random.seed(seed) # Python
        np.random.seed(seed) # NumPy/Pandas


    def _loaders(self, trainX, validX, trainY, validY):
        """Training and validation loaders, built once so the warm-up and the training that
        follows it go through the very same pipeline."""
        if self.__loaders is None:
            self.__loaders = (
                self.dl_factory.fromNumpy(trainX, trainY, self.batch_size, shuffle=True, seed=self.seed),
                self.dl_factory.fromNumpy(validX, validY, self.batch_size, shuffle=False)
            )

        return self.__loaders


    def __warmup_batches(self, X, Y):
        """The batches to warm up one graph with: PRECOMPILE_STEPS full-size ones, plus a smaller
        one when the split does not divide evenly, since that remainder is a shape of its own
        that would otherwise get traced during the measured training.

        Drawn from a throwaway loader instead of the real one, so the real loader's shuffling is
        not advanced and the first epoch still sees the batches it would have without warming up.
        """
        samples = self.PRECOMPILE_STEPS * self.batch_size
        batches = list(self.dl_factory.fromNumpy(X[:samples], Y[:samples], self.batch_size, shuffle=False))

        remainder = len(X) % self.batch_size

        if remainder:
            batches += list(self.dl_factory.fromNumpy(X[:remainder], Y[:remainder], remainder, shuffle=False))

        return batches


    def precompile(self, trainX, validX, trainY, validY):
        """Runs a few training and evaluation steps so the framework compiles its graphs and warms
        up its kernels before the measured training starts.

        Each runner restores afterwards whatever its steps changed (weights, optimizer state and
        random streams), so the training that follows is the one that would have run without it.
        """
        # Built here, so their one-off cost lands outside the measured training too
        self._loaders(trainX, validX, trainY, validY)

        self._precompile(self.__warmup_batches(trainX, trainY), self.__warmup_batches(validX, validY))


    def train(self, trainX, validX, trainY, validY):
        return self._train(*self._loaders(trainX, validX, trainY, validY))


    @abstractmethod
    def define_model(self):
        pass

    @abstractmethod
    def _precompile(self, train_batches, val_batches):
        pass

    @abstractmethod
    def _train(self, train_dl, val_dl):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def evaluate(self, testX, testY):
        pass
