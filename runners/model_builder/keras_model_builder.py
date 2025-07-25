
from keras import layers
from keras.layers import (LSTM, BatchNormalization, Conv2D,
                          Dense, Dropout, Flatten,
                          Input, MaxPooling2D, Activation)
from keras.models import Sequential
from keras.optimizers import Adam

from runners.model_builder.model_builder import ModelBuilder
from runners.model_builder.models.keras.mlp_complex import MLPComplex
from runners.model_builder.models.keras.cnn_complex import CNNComplex


class KerasModelBuilder(ModelBuilder):

    def _mlp_simple(self):
        activation = "relu"
        dropout = 0.2
        lr = 1e-4
        
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(784,)))

        # Hidden layer 1
        model.add(Dense(256, activation=activation))
        model.add(Dropout(dropout))

        # Hidden layer 2
        model.add(Dense(128, activation=activation))
        model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(10, activation='softmax'))
        
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
        
        model = Sequential([
            Input(shape=(32, 32, 3)),
            Conv2D(filters=32, kernel_size=(3, 3), activation=activation, padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout),
            
            Conv2D(filters=32, kernel_size=(3, 3), activation=activation, padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout),
        
            Flatten(),
        
            Dense(128, activation=activation),
            Dropout(dropout),
        
            Dense(128, activation=activation),
            Dropout(dropout),
        
            # Output layer
            Dense(10, activation = "softmax")
        ])
        
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
        interval = 10
        window = 48 * 60 // interval
        
        cells = 32
        dropout = 0.1
        lr = 1e-4
        
        # Build the model
        model = Sequential([
            Input(shape=(window, 11)),

            LSTM(cells, return_sequences=True),
            BatchNormalization(),
            Dropout(dropout),

            LSTM(cells),
            BatchNormalization(),
            Dropout(dropout),

            Dense(16),
            BatchNormalization(),
            Activation("tanh"),
            Dropout(dropout),

            Dense(1) # Output (trip count)
        ])

        # Compile the model
        model.compile(
            optimizer = Adam(learning_rate = lr),             
            loss = 'mse',
            metrics = ['mae']
        )
        
        return model

    def _lstm_complex(self):
        interval = 10
        window = 48 * 60 // interval

        lstm_layers = 8
        cells = 512
        dropout = 0.4
        lr = 1e-4
        
        # Build the model
        model = Sequential()
        model.add(Input(shape=(window, 11))) # Input layer

        # Add the hidden LSTM layers
        for i in range(1, lstm_layers + 1):

            # From the middle, the cells are halved
            if (i > (lstm_layers // 2)):
                cells = cells // 2

            model.add(LSTM(cells, return_sequences = (i < lstm_layers))) # Last LSTM layer doesn't return sequences
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

            
        # Output layer
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))
        model.add(Dropout(0.2))

        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))
        model.add(Dropout(0.2))

        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))
        model.add(Dropout(0.1))
        
        model.add(Dense(1))

        # Compile the model
        model.compile(
            optimizer = Adam(learning_rate = lr),             
            loss = 'mse',
            metrics = ['mae']
        )
        
        return model


