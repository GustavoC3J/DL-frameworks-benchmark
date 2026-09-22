
from keras import layers, Sequential

from runners.model_builder.models.keras.initializers import Float32Orthogonal

# Same epsilon in the three frameworks: Keras defaults to 1e-3 and Flax to 1e-6
LN_EPSILON = 1e-5

def lstm_complex(cells, lstm_layers, dropout_rate=0.2):
    model = Sequential()

    # LSTM funnel
    for i in range(1, lstm_layers + 1):
        # Keras' own defaults, written out so torch and Flax can declare the same ones
        model.add(layers.LSTM(
            cells,
            return_sequences=(i < lstm_layers), # Last LSTM layer doesn't return sequences
            kernel_initializer="glorot_uniform",
            recurrent_initializer=Float32Orthogonal(),
            bias_initializer="zeros",
            unit_forget_bias=True
        ))
        model.add(layers.LayerNormalization(epsilon=LN_EPSILON))
        model.add(layers.Dropout(dropout_rate))
        
        # Cells are halved for the next layer
        cells = max(cells // 2, 64)

    # Dense funnel
    for units in [128, 64, 32]:
        model.add(layers.Dense(units, kernel_initializer="he_uniform"))
        model.add(layers.LayerNormalization(epsilon=LN_EPSILON))
        model.add(layers.Activation("relu"))
        model.add(layers.Dropout(dropout_rate))

    # Output layer
    model.add(layers.Dense(1, kernel_initializer="he_uniform"))
    model.add(layers.Activation("relu"))

    return model
