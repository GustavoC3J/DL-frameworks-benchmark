import keras
import numpy as np
from keras import ops
from keras.random import SeedGenerator

from runners.model_builder.keras_model_builder import KerasModelBuilder
from runners.runner import Runner
from utils.best_weights_callback import BestWeightsCallback
from utils.precision import get_keras_precision
from utils.stage_callback import StageCallback

# Keras backend each benchmark backend runs on (KERAS_BACKEND has to be set before importing keras)
KERAS_BACKENDS = {"tf-keras": "tensorflow", "torch-keras": "torch", "jax-keras": "jax"}


class KerasRunner(Runner):
    """tf-keras, torch-keras and jax-keras: the same Keras code, on the backend KERAS_BACKEND selects."""

    def __init__(self, backend, **kwargs):

        # A wrong KERAS_BACKEND would otherwise train silently on another backend
        expected = KERAS_BACKENDS[backend]
        if keras.backend.backend() != expected:
            raise RuntimeError(f"{backend} needs KERAS_BACKEND={expected}, but Keras is running on {keras.backend.backend()}")

        super().__init__(**kwargs)

        # Seeds python, numpy and the backend (TensorFlow or torch; JAX's is Keras' own)
        keras.utils.set_random_seed(self.seed)

        # Set global floating point precision
        keras.config.set_dtype_policy(get_keras_precision(self.precision))


    @property
    def data_framework(self):
        return "tf" if keras.backend.backend() == "tensorflow" else "torch"


    def define_model(self):
        self.model = KerasModelBuilder(self.model_type, self.model_complexity).build()

        # Keras-torch runs eagerly by default (tf and jax already use XLA).
        # One graph per batch shape, as in TorchRunner
        if keras.backend.backend() == "torch":
            import torch
            torch._dynamo.config.automatic_dynamic_shapes = False
            self.model.jit_compile = True


    def __seed_generators(self):
        """The SeedGenerator every random layer (dropout) owns. keras.utils.set_random_seed does not
        reset their counter, so the warm-up has to put it back where the training expects it."""
        generators = []

        # _flatten_layers is what Keras itself uses to walk nested layers
        for layer in self.model._flatten_layers():
            for attribute in vars(layer).values():
                if isinstance(attribute, SeedGenerator):
                    generators.append(attribute)

        return generators


    def _precompile(self, train_dl, val_dl):
        """Runs one fit() over the warm-up loaders, so the train and test functions are traced and
        compiled on the path the training takes, and then undoes everything it changed:
        weights, optimizer state, metrics and seeds."""

        # Builds the model without touching its weights
        # JAX cannot take a torch tensor directly -> np.asarray
        self.model(np.asarray(next(iter(train_dl))[0]))

        # Only what already exists is saved: the optimizer's slots appear on the first update and
        # start at zero, and this keeps values such as the learning rate intact
        optimizer = self.model.optimizer
        weights = self.model.get_weights()
        optimizer_state = {variable.path: ops.convert_to_numpy(variable) for variable in optimizer.variables}
        generators = self.__seed_generators()
        generator_states = [ops.convert_to_numpy(generator.state) for generator in generators]

        self.model.fit(train_dl, validation_data=val_dl, epochs=1, verbose=0)

        # Compiled functions are cached per signature, so restoring values does not discard them
        self.model.set_weights(weights)

        for variable in optimizer.variables:
            if variable.path in optimizer_state:
                variable.assign(optimizer_state[variable.path])
            else:
                # Slot created by the steps above (Adam's moments): it starts at zero
                variable.assign(ops.zeros(variable.shape, dtype=variable.dtype))

        for generator, state in zip(generators, generator_states):
            generator.state.assign(state)

        self.model.reset_metrics()


    def _sync(self):
        backend = keras.backend.backend()

        if backend == "tensorflow":
            import tensorflow as tf
            tf.test.experimental.sync_devices()

        elif backend == "torch":
            import torch
            torch.cuda.synchronize()

        else:
            # fit hands the JAX state back to the variables before validating
            import jax
            jax.block_until_ready([variable.value for variable in self.model.variables])


    def _train(self, train_dl, val_dl):
        # The best weights are kept in GPU memory and restored at the end of fit.
        # StageCallback goes first, so the validation stage ends before the copy of the best weights
        callbacks = [
            StageCallback(self.stage_monitor),
            BestWeightsCallback()
        ]

        # Train the model
        history = self.model.fit(
            train_dl,
            validation_data = val_dl,
            epochs = self.epochs,
            callbacks=callbacks
        )

        # Add epoch times
        history.history["epoch_time"] = callbacks[0].times

        return history.history


    def save(self, path):
        self.model.save(path + "/model.keras")


    def evaluate(self, testX, testY):
        test_dl = self.dl_factory.fromNumpy(testX, testY, self.batch_size, shuffle=False)

        return self.model.evaluate(test_dl)
