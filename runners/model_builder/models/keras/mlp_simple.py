
from keras import layers, Sequential

def mlp_simple(activation='relu', dropout_rate=0.2):

    return Sequential([
        layers.Dense(256, activation=activation, kernel_initializer="glorot_uniform"),
        layers.Dropout(dropout_rate),

        layers.Dense(128, activation=activation, kernel_initializer="glorot_uniform"),
        layers.Dropout(dropout_rate),
        
        layers.Dense(10, kernel_initializer="glorot_uniform") # softmax is applied in loss function
    ])
