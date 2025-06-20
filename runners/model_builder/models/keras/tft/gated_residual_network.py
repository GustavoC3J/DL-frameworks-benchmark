
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

    
    def build(self, input_shape):

        # Input dimension
        input_dim = input_shape[-1]

        Dense_wrapper = layers.TimeDistributed if self.time_distributed else lambda x: x

        # If input dimension doesn't match output's, apply a projection
        self.projection = None
        if input_dim != self.output_size:
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


    def call(self, inputs, context=None, training=None):

        # Residual connection
        residual = self.projection(inputs) if self.projection else inputs

        # First linear and add context (if available)
        x = self.dense1(inputs)

        if context:
            # If it's time series, add temporal dimension to context
            expanded_context = ops.expand_dims(context, axis=1) if self.time_distributed else context
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