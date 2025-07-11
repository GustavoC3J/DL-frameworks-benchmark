
from datasets.loader.dataset_loader.cifar10_loader import CIFAR10Loader
from datasets.loader.dataset_loader.fashion_mnist_loader import FashionMNISTLoader
from datasets.loader.dataset_loader.yellow_taxi_loader import YellowTaxiDatasetLoader


class DataLoader():

    def __init__(self, model_type, seed):
        self.model_type = model_type
        self.seed = seed

        self.dataset_map = {
            "mlp": FashionMNISTLoader(seed),
            "cnn": CIFAR10Loader(seed),
            "lstm": YellowTaxiDatasetLoader(seed),
        }

    def load_data(self, dataset_type, **kwargs):
        if self.model_type not in self.dataset_map:
            raise ValueError(f"Not supported: {self.model_type}")
        
        return self.dataset_map[self.model_type].load(dataset_type, **kwargs)

