import time
from energy_monitor import EnergyMonitor
from collections.abc import Callable


class StageMonitor:
    """
    Splits every epoch into its training and validation stages: times each one and delimits it as a
    codecarbon task, so that time and energy can be divided by the batches of each stage.
    """

    def __init__(self, sync: Callable[[], None], energy_monitor: EnergyMonitor | None = None):
        """
        sync: Waits for the work queued on the device, which runs asynchronously
        energy_monitor: EnergyMonitor of the training phase, if any
        """
        self.sync = sync
        self.energy_monitor = energy_monitor
        self.epoch = -1
        self._start = None
        self.times = {"train_time": [], "val_time": []}


    def __begin(self, stage: str):
        if self.energy_monitor is not None:
            self.energy_monitor.begin(stage, epoch=self.epoch)

        # The clock starts after switching the task, so its cost stays out of both stages
        self._start = time.time()


    def train_begin(self):
        self.epoch += 1
        self.__begin("train")


    def val_begin(self):
        self.sync()
        self.times["train_time"].append(time.time() - self._start)
        self.__begin("val")


    def epoch_end(self):
        self.sync()
        self.times["val_time"].append(time.time() - self._start)
