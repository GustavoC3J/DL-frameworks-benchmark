
from keras import layers, ops

class GLU(layers.Layer):

    def __init__(self, hidden_units, dropout_rate=None, time_distributed=True, activation=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.time_distributed = time_distributed
        self.activation = activation

        self.dropout = layers.Dropout(self.dropout_rate) if self.dropout_rate else None

        Dense_wrapper = layers.TimeDistributed if self.time_distributed else lambda x: x

        self.activation_layer = Dense_wrapper(layers.Dense(self.hidden_units, activation=self.activation))
        self.gated_layer = Dense_wrapper(layers.Dense(self.hidden_units, activation="sigmoid"))


    def build(self, input_shape):
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        x = self.dropout(inputs) if self.dropout else inputs
        
        activation_res = self.activation_layer(x)
        gated_res = self.gated_layer(x)

        x = ops.multiply(activation_res, gated_res)

        return x
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.hidden_units) if self.time_distributed else (input_shape[0], self.hidden_units)