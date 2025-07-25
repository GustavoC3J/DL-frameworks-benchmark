
from keras import layers, Sequential

def lstm_complex(cells, lstm_layers, dropout_rate=0.2):
    model = Sequential()
    
    # Add the hidden LSTM layers
    for i in range(1, lstm_layers + 1):
        # From the middle, the cells are halved
        if i > (lstm_layers // 2):
            cells = cells // 2

        model.add(layers.LSTM(cells, return_sequences=(i < lstm_layers))) # Last LSTM layer doesn't return sequences
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))

    # Output layer
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("tanh"))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("tanh"))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(64))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("tanh"))
    model.add(layers.Dropout(0.1))
    
    model.add(layers.Dense(1))

    return model
