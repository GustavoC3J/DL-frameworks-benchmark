
import GPUtil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import random
import time

from callbacks.TFMetricsCallback import TFMetricsCallback


class TFRunner:

    def __init__(self, model_type, model_complexity, epochs = 30, batch_size = 32, seed = 42):
        self.model_type = model_type

        if not(model_complexity == "simple" or model_complexity == "complex"):
            raise ValueError("Model complexity must be 'simple' or 'complex'")
        self.model_complexity = model_complexity

        self.epochs = epochs
        self.batch_size = batch_size
        
        # Fix seed
        self.seed = seed
        random.seed(seed) # Python
        np.random.seed(seed) # NumPy/Pandas
        tf.random.set_seed(seed) # TensorFlow


    def load_data(self, dataset_type):
        if (self.model_type == "mlp"):
            return self._load_fashion_mnist(dataset_type)
        elif (self.model_type == "cnn"):
            pass
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


    
    def define_model(self):

        if self.model_type == "mlp":
            self._define_mlp()

        elif self.model_type == "cnn":
            self._define_cnn()

        elif self.model_type == "lstm":
            self._define_lstm()

        else:
            raise ValueError("Model not supported")
        pass


    def _define_mlp(self):
        activation = "relu"
        dropout = 0.2
        lr = 1e-3

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


    def _define_cnn(self):
        pass


    def _define_lstm(self):
        pass


    def train(self):
        # Load data
        trainX, validX, trainY, validY = self.load_data("train")

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



