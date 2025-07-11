

from abc import ABC, abstractmethod


class DatasetLoader(ABC):
    def __init__(self, seed):
        self.seed = seed

    @abstractmethod
    def load(self, dataset_type, **kwargs):
        pass
