
import numpy as np
from keras import ops
from keras.random import SeedGenerator


def _seed_generators(model):
    """The SeedGenerator every random layer (dropout) owns. keras.utils.set_random_seed does not
    reset their counter, so the warm-up has to put it back where the training expects it."""
    generators = []

    # _flatten_layers is what Keras itself uses to walk nested layers
    for layer in model._flatten_layers():
        for attribute in vars(layer).values():
            if isinstance(attribute, SeedGenerator):
                generators.append(attribute)

    return generators


def _as_numpy(batches):
    """Keras' loader adapters hand the trainer numpy arrays: its JAX backend cannot take a torch
    tensor straight from the DataLoader."""
    return [(np.asarray(batch_x), np.asarray(batch_y)) for batch_x, batch_y in batches]


def precompile(model, train_batches, val_batches):
    """Builds and warms up the two functions fit() uses, train_function and test_function, and
    then undoes everything the steps changed: weights, optimizer state, metrics and seeds."""

    # test_function first: it builds the model without touching its weights
    for batch_x, batch_y in _as_numpy(val_batches):
        model.test_on_batch(batch_x, batch_y)

    # Only what already exists is saved: the optimizer's slots appear on the first update and
    # start at zero, and this keeps values such as the learning rate intact
    weights = model.get_weights()
    optimizer_state = {variable.path: ops.convert_to_numpy(variable) for variable in model.optimizer.variables}
    generators = _seed_generators(model)
    generator_states = [ops.convert_to_numpy(generator.state) for generator in generators]

    # train_function: forward, backward and optimizer update
    for batch_x, batch_y in _as_numpy(train_batches):
        model.train_on_batch(batch_x, batch_y)

    # Compiled functions are cached per signature, so restoring values does not discard them
    model.set_weights(weights)

    for variable in model.optimizer.variables:
        if variable.path in optimizer_state:
            variable.assign(optimizer_state[variable.path])
        else:
            # Slot created by the steps above (Adam's moments): it starts at zero
            variable.assign(ops.zeros_like(variable))

    for generator, state in zip(generators, generator_states):
        generator.state.assign(state)

    model.reset_metrics()
