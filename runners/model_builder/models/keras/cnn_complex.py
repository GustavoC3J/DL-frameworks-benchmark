
import keras
from keras import layers, ops, initializers
import numpy as np


class Block(keras.Layer):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride = 1,
        kernel_initializer = "he_uniform",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.conv1 = layers.Conv2D(out_channels, 3, padding='same', strides=stride, kernel_initializer=kernel_initializer, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        
        self.conv2 = layers.Conv2D(out_channels, 3, padding='same', kernel_initializer=kernel_initializer, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        self.downsample = None
        if stride > 1 or in_channels != out_channels:
            self.downsample = layers.Conv2D(out_channels, kernel_size=1, padding='same', strides=stride, kernel_initializer=kernel_initializer, use_bias=False)

        # Config for model saving
        self.config = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "stride": stride,
            "kernel_initializer": kernel_initializer
        }

    
    def call(self, x, training = False):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = ops.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        if self.downsample:
            residual = self.downsample(residual)
        
        x += residual
        x = ops.relu(x)

        return x
    
    def get_config(self):
        super_config = super().get_config()
        super_config.update(self.config)
        return super_config





class CNNComplex(keras.Model):

    def __init__(
        self,
        n_blocks,
        starting_channels = 16,
        kernel_initializer = "he_uniform",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model_layers = []
        
        self.model_layers.append(layers.Conv2D(starting_channels, 3, padding='same', kernel_initializer=kernel_initializer, use_bias=False))
        self.model_layers.append(layers.BatchNormalization())
        self.model_layers.append(layers.ReLU())

        # Each stage is composed of n blocks whose convolutions use the corresponding filters
        filters = [16, 32, 64]
        in_channels = starting_channels

        for stage, out_channels in enumerate(filters):
            for i in range(n_blocks):
                # If it is the first block of the stage, a subsampling is made
                stride = 2 if stage > 0 and i == 0 else 1
                
                self.model_layers.append(Block(in_channels, out_channels, stride, kernel_initializer))
                in_channels = out_channels

        # Flatten and perform final prediction
        self.model_layers.append(layers.GlobalAveragePooling2D())
        self.model_layers.append(layers.Dense(10, activation = "softmax"))


        # Config for model saving
        self.config = {
            "n_blocks": n_blocks,
            "starting_channels": starting_channels,
            "kernel_initializer": kernel_initializer
        }


    def call(self, inputs, training = False):
        x = inputs

        for layer in self.model_layers:
            # Check if the "training" parameter is needed
            if isinstance(layer, (Block, layers.BatchNormalization)):
                x = layer(x, training=training)
            else:
                x = layer(x)

        return x
    
    
    def get_config(self):
        super_config = super().get_config()
        super_config.update(self.config)
        return super_config



