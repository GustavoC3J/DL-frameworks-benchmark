
from keras import layers, Sequential

def lstm_complex(cells, lstm_layers, dropout_rate=0.2):
    model = Sequential()
    
    # LSTM funnel
    for i in range(1, lstm_layers + 1):
        model.add(layers.LSTM(cells, return_sequences=(i < lstm_layers))) # Last LSTM layer doesn't return sequences
        model.add(layers.LayerNormalization())
        model.add(layers.Dropout(dropout_rate))
        
        # Cells are halved for the next layer
        cells = max(cells // 2, 64)

    # Dense funnel
    model.add(layers.Dense(128, kernel_initializer="he_uniform"))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(64, kernel_initializer="he_uniform"))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(32, kernel_initializer="he_uniform"))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(1, kernel_initializer="he_uniform"))
    model.add(layers.Activation("relu"))

    return model
