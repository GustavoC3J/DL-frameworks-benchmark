
from keras import layers, Sequential

def mlp_simple(activation='relu', dropout_rate=0.2):

    return Sequential([
        layers.Dense(256, activation=activation),
        layers.Dropout(dropout_rate),

        layers.Dense(128, activation=activation),
        layers.Dropout(dropout_rate),
        
        layers.Dense(10, activation='softmax')
    ])
