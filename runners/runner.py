import random
from abc import ABC, abstractmethod

import numpy as np

from datasets.loader.data_loader_factory import DataLoaderFactory


class Runner(ABC):

    # Data pipeline of the backend: tf.data in TensorFlow, torch's DataLoader everywhere else
    data_framework = None

    # Batches each graph is warmed up with before the measured training starts
    PRECOMPILE_STEPS = 2

    def __init__(self, model_type, model_complexity, keras, epochs, batch_size, seed, gpu_ids, precision):

        self.model_type = model_type
        self.keras = keras
        self.model_complexity = model_complexity
        self.epochs = epochs
        self.batch_size = batch_size
        self.gpu_ids = [int(gpu) for gpu in gpu_ids.split(",") if gpu.isdigit()]
        self.precision = precision

        self.dl_factory = DataLoaderFactory(self.data_framework)
        self.__loaders = None
        
        # Fix seed
        self.seed = seed
        random.seed(seed) # Python
        np.random.seed(seed) # NumPy/Pandas

        if self.data_framework == "torch":
            # The DataLoader shuffles with torch's global generator, seeded at random otherwise
            import torch
            torch.manual_seed(seed)


    def _loaders(self, trainX, validX, trainY, validY):
        """Training and validation loaders, built once so the warm-up and the training that
        follows it go through the very same pipeline."""
        if self.__loaders is None:
            self.__loaders = (
                self.dl_factory.fromNumpy(trainX, trainY, self.batch_size, shuffle=True),
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


    def __rng_state(self):
        """The random streams shared by the whole process. The loaders draw from them to shuffle,
        so the warm-up has to leave them exactly where the training expects them."""
        state = {"python": random.getstate(), "numpy": np.random.get_state()}

        if self.data_framework == "torch":
            import torch
            state["torch"] = torch.get_rng_state()

        return state


    def __restore_rng(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])

        if "torch" in state:
            import torch
            torch.set_rng_state(state["torch"])


    def precompile(self, trainX, validX, trainY, validY):
        """Runs a few training and evaluation steps so the framework compiles its graphs and warms
        up its kernels before the measured training starts.

        Weights, optimizer state and random streams are restored afterwards, so the training that
        follows is exactly the one that would have run without warming up.
        """
        # Built here, so their one-off cost lands outside the measured training too
        self._loaders(trainX, validX, trainY, validY)

        rng_state = self.__rng_state()

        self._precompile(self.__warmup_batches(trainX, trainY), self.__warmup_batches(validX, validY))

        self.__restore_rng(rng_state)


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
