
from keras import layers, Sequential

def lstm_simple(cells, dropout_rate=0.2):

    return Sequential([
        layers.LSTM(cells, return_sequences=True),
        layers.Dropout(dropout_rate),

        layers.LSTM(cells),
        layers.Dropout(dropout_rate),

        layers.Dense(cells // 2),
        layers.Activation("tanh"),
        layers.Dropout(dropout_rate),

        layers.Dense(1) # Output (trip count)
    ])
