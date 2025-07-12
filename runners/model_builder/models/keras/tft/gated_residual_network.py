
from keras import layers, ops

from runners.model_builder.models.keras.tft.glu import GLU

class GatedResidualNetwork(layers.Layer):
    """Gated Residual Network (GRN)"""

    def __init__(self, hidden_units, output_size=None, dropout_rate=None, time_distributed=True, **kwargs):
        super().__init__(**kwargs)

        self.hidden_units = hidden_units
        self.output_size = output_size if output_size != None else hidden_units
        self.dropout_rate = dropout_rate
        self.time_distributed = time_distributed

        Dense_wrapper = layers.TimeDistributed if self.time_distributed else lambda x: x

        self.projection = Dense_wrapper(layers.Dense(self.output_size))

        # Feed forward layers
        self.dense1 = Dense_wrapper(layers.Dense(self.hidden_units))
        self.context_dense = Dense_wrapper(layers.Dense(self.hidden_units, use_bias=False)) # No bias, to avoid two bias when added
        self.dense2 = Dense_wrapper(layers.Dense(self.hidden_units))

        # Gating layer
        self.glu = GLU(self.output_size, self.dropout_rate, self.time_distributed)

        # Normalization
        self.layer_norm = layers.LayerNormalization()

        # Dropout
        self.dropout = layers.Dropout(self.dropout_rate) if self.dropout_rate else lambda x: x


    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, context=None, training=None):

        # Residual connection
        # If input dimension doesn't match output's, apply a projection
        residual = self.projection(inputs) if inputs.shape[-1] != self.output_size else inputs

        # First linear and add context (if available)
        x = self.dense1(inputs)

        if context is not None:
            # If it's time series, add temporal dimension to context
            if self.time_distributed:
                expanded_context = ops.expand_dims(context, axis=1)
                expanded_context = ops.repeat(expanded_context, repeats=x.shape[1], axis=1)
            else:
                expanded_context = context

            context_transformed = self.context_dense(expanded_context)
            x = ops.add(x, context_transformed)

        # Activation, second linear and dropout (if available)
        x = ops.elu(x)
        x = self.dense2(x)
        x = self.dropout(x) # Keras passes training arg to dropout

        # Gating (GLU)
        x = self.glu(x)

        # Residual connection + normalization
        x = ops.add(x, residual)
        x = self.layer_norm(x)

        return x
    
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_size) if self.time_distributed else (input_shape[0], self.output_size)