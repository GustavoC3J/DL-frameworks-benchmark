
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(model_type, dataset_type, seed = 42):
    if (model_type == "mlp"):
        return _load_fashion_mnist(dataset_type, seed)
    
    elif (model_type == "cnn"):
        return _load_cifar_100(dataset_type, seed)
    
    else:
        pass


def _load_fashion_mnist(dataset_type, seed):

    if (dataset_type == "train"):
        # Load the training dataset
        data = pd.read_csv('datasets/fashion-mnist/fashion-mnist_train.csv')

        # Extract labels and pixel values
        labels = data['label']
        images = data.drop('label', axis=1).values

        # Scale the images between 0 and 1
        images = images / 255.0

        # Split into training and validation sets (80%-20%)
        # Returns trainX, validX, trainY, validY 
        return train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=seed)

    else:
        # Load the test dataset
        test = pd.read_csv('datasets/fashion-mnist/fashion-mnist_test.csv')

        # Extract labels and pixel values
        testY = test['label']
        testX = test.drop('label', axis=1).values / 255.0

        return (testX, testY)
    
def _load_cifar_100(dataset_type, seed):
    path = f"datasets/cifar-100/{dataset_type}"

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    def shape_images(img):
        num_images = img.shape[0]
        
        # Split into three channels
        images = img.reshape(num_images, 3, 1024)
        
        # Reshape to image width and height
        images = images.reshape(num_images, 3, 32, 32)
        
        # Change order of channels (num_imagenes, 32, 32, 3)
        images = images.transpose(0, 2, 3, 1)

        return images

    # Load data
    dataset = unpickle(path)
    images = dataset[b'data']
    labels = np.array(dataset[b'coarse_labels'])

    # Prepare data
    images = shape_images(images)
    images = images / 255.0 # Scale the images between 0 and 1

    if (dataset_type == "train"):
        # Split into training and validation sets (80%-20%)
        # Returns trainX, validX, trainY, validY 
        return train_test_split(images, labels, test_size = 0.2, stratify = labels, random_state = seed)
    else:
        return (images, labels)