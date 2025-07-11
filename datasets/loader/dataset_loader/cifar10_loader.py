
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

from datasets.loader.dataset_loader.dataset_loader import DatasetLoader


class CIFAR10Loader(DatasetLoader):

    def load(self, dataset_type, **kwargs):

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
        path = f"datasets/cifar10/cifar-10-batches-py"

        if (dataset_type == "train"):
            images = []
            labels = []

            # Join all files into numpy arrays
            for i in range(1, 6):
                batch = unpickle(f"{path}/data_batch_{i}")
                images.append(batch[b'data'])
                labels.extend(batch[b'labels'])

            images = np.vstack(images)
            labels = np.array(labels)

        else:
            batch = unpickle(f"{path}/test_batch")
            images = batch[b'data']
            labels = np.array(batch[b'labels'])
        
        # Prepare data
        images = shape_images(images)
        images = images / 255.0 # Scale the images between 0 and 1

        if dataset_type == "train":
            # Split into training and validation sets (80%-20%)
            # Returns trainX, validX, trainY, validY 
            return train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=self.seed)
        else:
            return images, labels
