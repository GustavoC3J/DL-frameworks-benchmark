
from keras.optimizers import Adam

from runners.model_builder.model_builder import ModelBuilder
from runners.model_builder.models.keras.cnn_complex import CNNComplex
from runners.model_builder.models.keras.cnn_simple import cnn_simple
from runners.model_builder.models.keras.lstm_complex import lstm_complex
from runners.model_builder.models.keras.lstm_simple import lstm_simple
from runners.model_builder.models.keras.mlp_complex import MLPComplex
from runners.model_builder.models.keras.mlp_simple import mlp_simple


class KerasModelBuilder(ModelBuilder):

    def _mlp_simple(self):
        activation = "relu"
        dropout = 0.2
        lr = 1e-4
        
        model = mlp_simple(activation, dropout)
        
        # Compile the model
        model.compile(
            optimizer = Adam(learning_rate = lr),             
            loss = 'sparse_categorical_crossentropy',
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
            loss = 'sparse_categorical_crossentropy',
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
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
        )

        return model
    

    def _cnn_complex(self):
        lr = 1e-3

        # Build the model
        model = CNNComplex(
            n_blocks=5, # number of blocks, 6n + 2 layers
            starting_channels=64,
            kernel_initializer="he_uniform"
        )
        
        # Compile the model
        model.compile(
            optimizer = Adam(learning_rate = lr),             
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
        )

        return model
    

    def _lstm_simple(self):        
        cells = 16
        dropout = 0.1
        lr = 1e-4
        
        # Build the model
        model = lstm_simple(cells, dropout)

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
        model = lstm_complex(cells, lstm_layers, dropout_rate=dropout)

        # Compile the model
        model.compile(
            optimizer = Adam(learning_rate = lr),             
            loss = 'mse',
            metrics = ['mae']
        )
        
        return model


