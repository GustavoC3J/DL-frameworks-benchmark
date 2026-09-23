
from keras import layers, Sequential

def cnn_simple(activation='relu', dropout_rate=0.2):

    return Sequential([
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation=activation, padding='same', kernel_initializer="glorot_uniform"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(dropout_rate),
        
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation=activation, padding='same', kernel_initializer="glorot_uniform"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(dropout_rate),
    
        layers.Flatten(),
    
        layers.Dense(128, activation=activation, kernel_initializer="glorot_uniform"),
        layers.Dropout(dropout_rate),
    
        layers.Dense(128, activation=activation, kernel_initializer="glorot_uniform"),
        layers.Dropout(dropout_rate),
    
        # Output layer
        layers.Dense(100, activation = "softmax", kernel_initializer="glorot_uniform")
    ])
