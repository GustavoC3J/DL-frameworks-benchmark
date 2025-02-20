

from keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, GlobalAveragePooling2D,
                                     Input, MaxPooling2D, ReLU, add, LSTM, Bidirectional)
from keras.models import Model, Sequential
from keras.optimizers import Adam

from runners.model_builder.model_builder import ModelBuilder


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
        activation = "relu"
        dropout = 0.2
        lr = 1e-4

        hidden_layers = 21
        final_units = 128  # Last hidden layers will have 128 units
        layers_per_group = 3

        # Calculate initial units based on number of hidden layers
        groups = (hidden_layers + layers_per_group - 1) // layers_per_group
        units = final_units * (2 ** (groups - 1)) # Starting units

        
        model = Sequential()
        model.add(Input(shape=(784,))) # Input layer

        # Add the hidden layers
        for i in range(1, hidden_layers + 1):
            model.add(Dense(units, activation=activation))
            model.add(Dropout(dropout))

            # Halve the number of units for the next group
            if i % layers_per_group == 0:
                units //= 2
                
        # Output layer
        model.add(Dense(10, activation='softmax'))

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
            Dense(20, activation = "softmax")
        ])
        
        # Compile the model
        model.compile(
            optimizer = Adam(learning_rate = lr),             
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
        )

        return model
    

    def _cnn_complex(self):
        n = 10 # number of blocks, 6n + 1 layers
        lr = 1e-4

        # Build a Resnet block
        def block(x, filtros, kernel_size = 3, stride = 1):
            residual = x

            x = Conv2D(filters = filtros, kernel_size = kernel_size, padding = 'same', strides = stride)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            x = Conv2D(filters = filtros, kernel_size = kernel_size, padding = 'same')(x)
            x = BatchNormalization()(x)

            if stride > 1:
                residual = Conv2D(filters = filtros, kernel_size = 1, padding = 'same', strides = stride)(residual)
            
            x = add([x, residual])
            x = ReLU()(x)

            return x
        
        # Initial layer
        input = Input(shape = (32, 32, 3))
        x = Conv2D(filters = 16, kernel_size = 3, padding = 'same')(input)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Each stage is composed of n blocks whose convolutions use the corresponding filters
        for stage, filters in enumerate([16, 32, 64]):
            for i in range(n):
                # If it is the first block of the stage, a subsampling is made
                stride = 2 if stage > 0 and i == 0 else 1
                
                x = block(x, filters, stride = stride)

        # Flatten and perform final prediction
        x = GlobalAveragePooling2D()(x)
        x = Dense(20, activation = "softmax")(x)
        

        # Build the model
        model = Model(inputs = input, outputs = x)
        
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
        activation = "tanh"
        dropout = 0.1
        lr = 1e-4
        
        # Build the model
        model = Sequential([
            Input(shape=(window, 11)),

            LSTM(cells, activation=activation, return_sequences=True),
            BatchNormalization(),
            Dropout(dropout),

            LSTM(cells, activation=activation),
            BatchNormalization(),
            Dropout(dropout),

            Dense(16, activation = activation),
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
        activation = "tanh"
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

            model.add(Bidirectional(LSTM(cells, activation = activation, return_sequences = (i < lstm_layers)))) # Last LSTM layer doesn't return sequences
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

            
        # Output layer
        model.add(Dense(256, activation = activation))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(128, activation = activation))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(Dense(64, activation = activation))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        
        model.add(Dense(1))

        # Compile the model
        model.compile(
            optimizer = Adam(learning_rate = lr),             
            loss = 'mse',
            metrics = ['mae']
        )
        
        return model


