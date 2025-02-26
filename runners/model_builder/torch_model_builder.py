
import torch.nn as nn
import torch.optim as optim


from runners.model_builder.model_builder import ModelBuilder
from utils.torch_utils import accuracy


class TorchModelBuilder(ModelBuilder):

    def _mlp_simple(self):
        activation = nn.ReLU()
        dropout = 0.2
        lr = 1e-4
        
        model = nn.Sequential(
            nn.Linear(784, 256),
            activation,
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            activation,
            nn.Dropout(dropout),

            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

        config = {
            "optimizer": optim.Adam(model.parameters(), lr=lr),
            "loss_fn": nn.CrossEntropyLoss(),
            "metric_fn": accuracy,
            "metric_name": "accuracy"
        }

        return model, config
    

    def _mlp_complex(self):
        raise NotImplementedError()


    def _cnn_simple(self):
        """
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
        """


    def _cnn_complex(self):
        raise NotImplementedError()
    

    def _lstm_simple(self):
        """
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
        """

    def _lstm_complex(self):
        raise NotImplementedError()