from keras import ops
from keras.callbacks import Callback


class BestWeightsCallback(Callback):
    """
    Keeps a copy of the weights with the lowest val_loss and restores them when 
    training ends.

    The copy is taken using ops, so that the tensors are kept on the GPU.
    """

    def on_train_begin(self, logs=None):
        self.best_val_loss = float("inf")
        self.best_weights = None
        self.best_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        val_loss = float(logs["val_loss"])

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.best_weights = [ops.copy(v.value) for v in self.model.weights]

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            for variable, value in zip(self.model.weights, self.best_weights):
                variable.assign(value)
