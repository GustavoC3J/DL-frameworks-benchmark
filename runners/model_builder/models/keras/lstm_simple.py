
from keras import layers, Sequential

from runners.model_builder.models.keras.initializers import Float32Orthogonal

def lstm_simple(cells, dropout_rate=0.2):

    # Keras' own defaults, written out so torch and Flax can declare the same ones.
    # A new initializer per layer: an instance fixes its seed and would repeat the kernel
    lstm_init = lambda: dict(
        kernel_initializer="glorot_uniform",
        recurrent_initializer=Float32Orthogonal(),
        bias_initializer="zeros",
        unit_forget_bias=True
    )

    return Sequential([
        layers.LSTM(cells, return_sequences=True, **lstm_init()),
        layers.Dropout(dropout_rate),

        layers.LSTM(cells, **lstm_init()),
        layers.Dropout(dropout_rate),

        layers.Dense(cells // 2, kernel_initializer="glorot_uniform"),
        layers.Activation("relu"),
        layers.Dropout(dropout_rate),

        layers.Dense(1, kernel_initializer="glorot_uniform"), # Output (trip count)
        layers.Activation("relu")
    ])
