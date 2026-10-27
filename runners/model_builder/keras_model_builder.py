
import keras
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

from runners.model_builder.model_builder import ModelBuilder
from runners.model_builder.models.keras.cnn_complex import CNNComplex
from runners.model_builder.models.keras.cnn_simple import cnn_simple
from runners.model_builder.models.keras.lstm_complex import lstm_complex
from runners.model_builder.models.keras.lstm_simple import lstm_simple
from runners.model_builder.models.keras.mlp_complex import MLPComplex
from runners.model_builder.models.keras.mlp_simple import mlp_simple
from runners.model_builder.models.keras.transformer import Transformer


class KerasModelBuilder(ModelBuilder):
    """The classification models output logits, as torch and Flax do: from probabilities, the
    torch and JAX backends clip them to 1e-7 and the loss saturates, while TensorFlow's does not."""

    def __init__(self, model_type, model_complexity, jax_lstm_cudnn=False):
        super().__init__(model_type, model_complexity)
        self.jax_lstm_cudnn = jax_lstm_cudnn


    def _lstm_cudnn(self):
        # Keras' "auto" only falls back when cuDNN fails while tracing, but JAX's cuDNN LSTM rejects
        # anything but float32 when compiling. True instead of "auto": fail rather than fall back
        if keras.backend.backend() == "jax":
            return self.jax_lstm_cudnn
        return "auto"


    def _disable_jax_attention_cudnn(self):
        # JAX's cuDNN attention (fp16/bf16 only) leaves a CUDA error that breaks the next compilation.
        # Global switch: the layer turns flash_attention=False into the global default (None, auto)
        if keras.backend.backend() == "jax":
            keras.config.disable_flash_attention()


    def _mlp_simple(self):
        activation = "relu"
        dropout = 0.2
        lr = 1e-4
        
        model = mlp_simple(activation, dropout)
        
        # Compile the model
        model.compile(
            optimizer = Adam(learning_rate = lr),             
            loss = SparseCategoricalCrossentropy(from_logits=True),
            metrics = ['accuracy']
        )

        return model
    

    def _mlp_complex(self):
        lr = 3e-5

        model = MLPComplex(
            groups=5,
            layers_per_group=2,
            dropout=0.2,
            initial_units=800,
            final_units=160,
            kernel_initializer="he_uniform"
        )

        # Compile the model
        model.compile(
            optimizer = Adam(learning_rate = lr),             
            loss = SparseCategoricalCrossentropy(from_logits=True),
            metrics = ['accuracy']
        )

        return model


    def _cnn_simple(self):
        activation = "relu"
        dropout = 0.2
        lr = 1e-4
        
        model = cnn_simple(activation, dropout)
        
        # Compile the model
        model.compile(
            optimizer = Adam(learning_rate = lr),             
            loss = SparseCategoricalCrossentropy(from_logits=True),
            metrics = ['accuracy']
        )

        return model
    

    def _cnn_complex(self):
        lr = 5e-3

        # Build the model
        model = CNNComplex(
            n_blocks=5, # number of blocks, 6n + 2 layers
            starting_channels=64,
            filters=(64, 128, 256), # 4 times ResNet32's widths
            kernel_initializer="he_uniform"
        )
        
        # Compile the model
        model.compile(
            optimizer = Adam(learning_rate = lr),             
            loss = SparseCategoricalCrossentropy(from_logits=True),
            metrics = ['accuracy']
        )

        return model
    

    def _lstm_simple(self):        
        cells = 16
        dropout = 0.1
        lr = 1e-4
        
        # Build the model
        model = lstm_simple(cells, dropout, use_cudnn=self._lstm_cudnn())

        # Compile the model
        model.compile(
            optimizer = Adam(learning_rate = lr),             
            loss = 'mse',
            metrics = ['mae']
        )
        
        return model

    def _lstm_complex(self):
        lstm_layers = 3
        cells = 512
        dropout = 0.1
        lr = 1e-4
        
        # Build the model
        model = lstm_complex(cells, lstm_layers, dropout_rate=dropout, use_cudnn=self._lstm_cudnn())

        # Compile the model
        model.compile(
            optimizer = Adam(learning_rate = lr),
            loss = 'mse',
            metrics = ['mae']
        )

        return model


    def _transformer_simple(self):
        interval = 10
        window = 24 * 60 // interval
        horizon = 12 # Steps of trip count to predict

        lr = 1e-4

        # Build the model
        self._disable_jax_attention_cudnn()
        model = Transformer(
            d_model=128,
            num_heads=4,
            d_ff=256,
            encoder_layers=1,
            decoder_layers=1,
            dropout=0.1,
            source_length=window,
            horizon=horizon
        )

        # Compile the model
        model.compile(
            optimizer = Adam(learning_rate = lr),
            loss = 'mse',
            metrics = ['mae']
        )

        return model


    def _transformer_complex(self):
        interval = 10
        window = 24 * 60 // interval
        horizon = 12

        lr = 1e-4

        self._disable_jax_attention_cudnn()
        model = Transformer(
            d_model=512,
            num_heads=8,
            d_ff=2048,
            encoder_layers=6,
            decoder_layers=6,
            dropout=0.1,
            source_length=window,
            horizon=horizon
        )

        model.compile(
            optimizer = Adam(learning_rate = lr),
            loss = 'mse',
            metrics = ['mae']
        )

        return model
