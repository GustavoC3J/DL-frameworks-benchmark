import time

from keras.callbacks import Callback
from stage_monitor import StageMonitor


class StageCallback(Callback):
    """
    Hands the epochs of fit to a StageMonitor and times each whole epoch. on_test_begin only fires
    for the validation of fit, because the test is evaluated without callbacks.
    """

    def __init__(self, stage_monitor: StageMonitor):
        super().__init__()
        self.stage_monitor = stage_monitor
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.stage_monitor.train_begin()

    def on_test_begin(self, logs=None):
        self.stage_monitor.val_begin()

    def on_epoch_end(self, epoch, logs=None):
        self.stage_monitor.epoch_end()
        self.times.append(time.time() - self.epoch_start_time)
