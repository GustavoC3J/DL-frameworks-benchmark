import numpy as np
import pickle
from sklearn.model_selection import train_test_split

from datasets.loader.dataset_loader.dataset_loader import DatasetLoader

PATH = "datasets/cifar100/cifar-100-python"


class CIFAR100Loader(DatasetLoader):

    def load(self, dataset_type, **kwargs):
        file = "train" if dataset_type == "train" else "test"

        with open(f"{PATH}/{file}", "rb") as fo:
            batch = pickle.load(fo, encoding="bytes")

        # Fine labels: the 100 classes, not the 20 superclasses
        labels = np.array(batch[b"fine_labels"])

        # Rows are stored channel-first (3 × 1024), so reorder to NHWC
        images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        # Scale the images between 0 and 1. float32 is what the tensors use: halves the memory
        images = (images / 255.0).astype("float32")

        if dataset_type == "train":
            # Split into training and validation sets (80%-20%)
            # Returns trainX, validX, trainY, validY
            return train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=self.seed)
        else:
            return images, labels
