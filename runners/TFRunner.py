
import pickle
import GPUtil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, ReLU, GlobalAveragePooling2D, add
from tensorflow.keras.optimizers import Adam

import random
import time

from utils.TFMetricsCallback import TFMetricsCallback


class TFRunner:

    def __init__(self, model_type, model_complexity, epochs = 30, batch_size = 32, gpus = ["GPU:0"], seed = 42, n = 10):
        self.model_type = model_type

        if not(model_complexity == "simple" or model_complexity == "complex"):
            raise ValueError("Model complexity must be 'simple' or 'complex'")
        self.model_complexity = model_complexity

        self.epochs = epochs
        self.batch_size = batch_size
        self.gpus = gpus
        
        # Fix seed
        self.seed = seed
        random.seed(seed) # Python
        np.random.seed(seed) # NumPy/Pandas
        tf.random.set_seed(seed) # TensorFlow

        # CNN
        self.n = n


    def load_data(self, dataset_type):
        if (self.model_type == "mlp"):
            return self._load_fashion_mnist(dataset_type)
        
        elif (self.model_type == "cnn"):
            return self._load_cifar_100(dataset_type)
        
        else:
            pass

    
    def _load_fashion_mnist(self, dataset_type):

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
            return train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=self.seed)

        else:
            # Load the test dataset
            test = pd.read_csv('datasets/fashion-mnist/fashion-mnist_test.csv')

            # Extract labels and pixel values
            testY = test['label']
            testX = test.drop('label', axis=1).values / 255.0

            return (testX, testY)
        
    def _load_cifar_100(self, dataset_type):
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
            return train_test_split(images, labels, test_size = 0.2, stratify = labels, random_state = self.seed)
        else:
            return (images, labels)




    
    def define_model(self):

        if len(self.gpus) > 1:
            # Define the strategy to follow in order to balance the workload between GPUs
            strategy = tf.distribute.MirroredStrategy(self.gpus)
        else:
            strategy = tf.distribute.get_strategy()


        if self.model_type == "mlp":
            self._define_mlp(strategy)

        elif self.model_type == "cnn":
            self._define_cnn(strategy)

        elif self.model_type == "lstm":
            self._define_lstm(strategy)

        else:
            raise ValueError("Model not supported")
        pass


    def _define_mlp(self, strategy):
        activation = "relu"
        dropout = 0.2
        lr = 1e-3

        with strategy.scope():
            self.model = Sequential()
            self.model.add(Input(shape=(784,))) # Input layer


            if(self.model_complexity == "simple"):

                # Input + Hidden layer 1
                self.model.add(Dense(256, activation=activation))
                self.model.add(Dropout(dropout))

                # Hidden layer 2
                self.model.add(Dense(128, activation=activation))
                self.model.add(Dropout(dropout))
                
            else:
                hidden_layers = 21
                final_units = 128  # Last hidden layers will have 128 units
                layers_per_group = 3

                # Calculate initial units based on number of hidden layers
                groups = (hidden_layers + layers_per_group - 1) // layers_per_group
                units = final_units * (2 ** (groups - 1)) # Starting units

                # Add the rest of the hidden layers
                for i in range(1, hidden_layers + 1):
                    self.model.add(Dense(units, activation=activation))
                    self.model.add(Dropout(dropout))

                    # Halve the number of units for the next group
                    if i % layers_per_group == 0:
                        units //= 2  
            

            # Output layer
            self.model.add(Dense(10, activation='softmax'))
                
            # Compile the model
            self.model.compile(
                optimizer = Adam(learning_rate = lr),             
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy']
            )


    def _define_cnn(self, strategy):
        
        activation = "relu"
        dropout = 0.2
        lr = 1e-3

        with strategy.scope():

            if(self.model_complexity == "simple"):
                self.model = Sequential([
                    Conv2D(filters=32, kernel_size=(3, 3), activation=activation, padding='same', input_shape=(32, 32, 3)),
                    MaxPooling2D(pool_size=(2, 2)),
                    Dropout(dropout),
                    
                    Conv2D(filters=32, kernel_size=(3, 3), activation=activation, padding='same'),
                    MaxPooling2D(pool_size=(2, 2)),
                    Dropout(dropout),
                
                    Flatten(),
                
                    Dense(128, activation=activation),
                    Dropout(dropout),
                
                    Dense(128, activation=activation),
                    Dropout(dropout),
                
                    # Output layer
                    Dense(20, activation = "softmax")
                ])

            else:
                self._define_resnet(self.n)
            
            # Compile the model
            self.model.compile(
                optimizer = Adam(learning_rate = lr),             
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy']
            )

    def _define_resnet(self, n):

        # Build a Resnet block
        def block(x, filtros, kernel_size = 3, stride = 1):
            residual = x

            x = Conv2D(filters = filtros, kernel_size = kernel_size, padding = 'same', strides = stride)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = Conv2D(filters = filtros, kernel_size = kernel_size, padding = 'same')(x)
            x = BatchNormalization()(x)

            if stride > 1:
                residual = Conv2D(filters = filtros, kernel_size = 1, padding = 'same', strides = stride)(residual)
            
            x = add([x, residual])
            x = ReLU()(x)

            return x
        
        # Initial layer
        input = Input(shape = (32, 32, 3))
        x = Conv2D(filters = 16, kernel_size = 3, padding = 'same')(input)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Each stage is composed of n blocks whose convolutions use the corresponding filters
        for stage, filters in enumerate([16, 32, 64]):
            for i in range(n):
                # If it is the first block of the stage, a subsampling is made
                stride = 2 if stage > 0 and i == 0 else 1
                
                x = block(x, filters, stride = stride)

        # Flatten and perform final prediction
        x = GlobalAveragePooling2D()(x)
        x = Dense(20, activation = "softmax")(x)

        # Build the model
        self.model = Model(inputs = input, outputs = x)


    def _define_lstm(self, strategy):
        pass


    def train(self, trainX, validX, trainY, validY):
        
        # Train the model
        return self.model.fit(
            trainX,
            trainY,
            epochs = self.epochs,
            batch_size = self.batch_size,
            validation_data = (validX, validY),
            callbacks=[TFMetricsCallback()]
        )


    def evaluate(self):
        testX, testY = self.load_data("test")
        self.model.evaluate(testX, testY)



