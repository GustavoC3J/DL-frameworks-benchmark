from datasets.loader.dataset_loader.cifar10_loader import CIFAR10Loader
from datasets.loader.dataset_loader.cifar100_loader import CIFAR100Loader
from datasets.loader.dataset_loader.fashion_mnist_loader import FashionMNISTLoader
from datasets.loader.dataset_loader.yellow_taxi_loader import YellowTaxiDatasetLoader


DATASETS = {
    "fashion-mnist": FashionMNISTLoader,
    "cifar10": CIFAR10Loader,
    "cifar100": CIFAR100Loader,
    "yellow-taxi": YellowTaxiDatasetLoader,
}

DEFAULT_DATASET = {
    "mlp": "fashion-mnist",
    "cnn": "cifar10",
    "lstm": "yellow-taxi",
}


class DataLoader():

    def __init__(self, model_type, seed):
        self.model_type = model_type
        self.seed = seed

        if model_type not in DEFAULT_DATASET:
            raise ValueError(f"Not supported: {model_type}")
        dataset = DEFAULT_DATASET[model_type]

        self.dataset = dataset
        # A single instance: the taxi loader keeps the scaler fitted in train for the test set
        self.loader = DATASETS[dataset](seed)

    def load_data(self, dataset_type, **kwargs):
        return self.loader.load(dataset_type, **kwargs)
