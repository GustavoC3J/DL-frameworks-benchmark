
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets.loader.dataset_loader.dataset_loader import DatasetLoader


class FashionMNISTLoader(DatasetLoader):

    def load(self, dataset_type, **kwargs):
        if dataset_type == "train":
            # Load the training dataset
            data = pd.read_csv('datasets/fashion-mnist/fashion-mnist_train.csv')

            # Extract labels and pixel values
            labels = data['label'].values
            images = data.drop('label', axis=1).values

            # Scale the images between 0 and 1
            images = images / 255.0

            # Split into training and validation sets (80%-20%)
            # Returns trainX, validX, trainY, validY 
            return train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=self.seed)
        else:
            # Load the test dataset
            test = pd.read_csv('datasets/fashion-mnist/fashion-mnist_test.csv')

            # Extract labels and pixel values
            testY = test['label'].values
            testX = test.drop('label', axis=1).values / 255.0

            return (testX, testY)
        
