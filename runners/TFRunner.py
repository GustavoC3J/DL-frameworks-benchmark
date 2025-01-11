
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

import random
import time


class TFRunner:

    def __init__(self, model_type, model_complexity, seed = 42):
        self.model_type = model_type

        if not(model_complexity == "simple" or model_complexity == "complex"):
            raise ValueError("Model complexity must be 'simple' or 'complex'")
        self.model_complexity = model_complexity

        
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

        if(self.model_complexity == "simple"):
            activation = "relu"
            dropout = 0.2
            lr = 0.001

            self.model = Sequential([
                # Input + Hidden layer 1
                Dense(256, activation=activation, input_shape=(784,)),
                Dropout(dropout),
                
                # Hidden layer 2
                Dense(128, activation=activation),
                Dropout(dropout)
            ])
            
        else:
            activation = "relu"
            dropout = 0.2
            lr = 0.001
            
            self.model = Sequential([
                # Input + Hidden layer
                Dense(512, activation=activation, input_shape=(784)),
                Dropout(dropout),
                
                # Hidden layer
                Dense(512, activation=activation),
                Dropout(dropout),
                
                # Hidden layer
                Dense(256, activation=activation),
                Dropout(dropout),
                
                # Hidden layer
                Dense(256, activation=activation),
                Dropout(dropout),
                
                # Hidden layer
                Dense(128, activation=activation),
                Dropout(dropout)
            ])
        
        # Outut layer
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
        return self.model.fit(trainX, trainY, epochs = 30, validation_data = (validX, validY))


    def evaluate(self):
        testX, testY = self.load_data("test")
        self.model.evaluate(testX, testY)

