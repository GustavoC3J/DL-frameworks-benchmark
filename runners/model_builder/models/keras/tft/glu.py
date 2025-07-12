
from keras import layers, ops

class GLU(layers.Layer):

    def __init__(self, hidden_units, dropout_rate=None, time_distributed=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.time_distributed = time_distributed

        Dense = layers.TimeDistributed if time_distributed else lambda x: x
        self.dense = Dense(layers.Dense(hidden_units * 2))


    def build(self, input_shape):
        super().build(input_shape)
    

    def call(self, inputs, training=None):
        x = self.dropout(inputs) if self.dropout else inputs
        x = self.dense(x)
        return ops.glu(x)
    
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.hidden_units) if self.time_distributed else \
               (input_shape[0], self.hidden_units)