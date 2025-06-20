

from runners.model_builder.models.keras.tft.static_covariate_encoder import StaticCovariateEncoder
from runners.model_builder.models.keras.tft.variable_selection_network import VariableSelectionNetwork

import keras
from keras import layers, ops




#quitar metodos build

#Inputs

#Embeddings + Gated Residual Networks (GRN).

#Variable selection networks.

#LSTM (encoder/decoder).

#Self-attention temporal.

#Output layer.

class TFT(keras.Model):
    
    def __init__(self, hidden_units, output_size, dropout_rate=None, **kwargs):
        super().__init__(**kwargs)

        self.hidden_units = hidden_units
        self.output_size = output_size
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        #self.gnu = GRN(self.hidden_units, output_size=1)
        #self.vsn = VariableSelectionNetwork(self.hidden_units, dropout_rate=self.dropout_rate, time_distributed=True)
        self.sce = StaticCovariateEncoder(
            hidden_dim=self.hidden_units,
            num_static_vars=input_shape[-2],
            dropout_rate=self.dropout_rate
        )

        # resto de cosas

        #self.output_layer = layers.TimeDistributed(layers.Dense(self.output_size))
        
    def call(self, inputs):

        x = self.sce(inputs)

        #x = self.output_layer(x)

        return x
    