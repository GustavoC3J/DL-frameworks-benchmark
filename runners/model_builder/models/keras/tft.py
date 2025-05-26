
import keras
from keras import layers, ops


class GLU(layers.Layer):

    def __init__(self, hidden_units, dropout_rate=None, time_distributed=True, activation=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.time_distributed = time_distributed
        self.activation = activation

    def build(self, input_shape):
        self.dropout = layers.Dropout(self.dropout_rate) if self.dropout_rate else None

        Dense_wrapper = layers.TimeDistributed if self.time_distributed else lambda x: x

        self.activation_layer = Dense_wrapper(layers.Dense(self.hidden_units, activation=self.activation))
        self.gated_layer = Dense_wrapper(layers.Dense(self.hidden_units, activation="sigmoid"))

    
    def call(self, inputs, training=None):
        x = self.dropout(inputs) if self.dropout else inputs
        
        activation_res = self.activation_layer(x)
        gated_res = self.gated_layer(x)

        x = ops.multiply(activation_res, gated_res)

        return x
        


class GRN(layers.Layer):
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
        self.glu = GLU(self.hidden_units, self.dropout_rate, self.time_distributed)

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


class VariableSelectionNetwork(layers.Layer):
    """Variable Selection Network (VSN)"""

    def __init__(self, hidden_units, dropout_rate=None, time_distributed=True, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.time_distributed = time_distributed


    def build(self, input_shape):
        # input_shape = (batch_size, num_inputs, embedding_dim) or (batch_size, window, num_inputs, embedding_dim)
        embedding_dim = input_shape[-1]
        num_inputs = input_shape[-2]

        self.grns = [
            GRN(
                hidden_units,
                dropout_rate=self.dropout_rate,
                time_distributed=self.time_distributed
            )
            for _ in range(num_inputs)
        ]
        
        self.softmax = layers.Softmax()
        self.flatten = layers.Reshape( (input_shape[1], num_inputs * embedding_dim) if self.time_distributed else \
                                       (num_inputs * embedding_dim) )

        self.selection_grn = GRN(
            hidden_units,
            output_size=num_inputs,  # One for each feature
            dropout_rate=self.dropout_rate,
            time_distributed=self.time_distributed
        )

    def call(self, inputs, context=None, training=None):
        # inputs: (batch_size, num_inputs, embedding_dim) or (batch_size, window, num_inputs, embedding_dim)

        # Variable selection weights
        flatten_inputs = self.flatten(inputs)  # (batch_size, window, num_inputs * embedding_dim)
        weights = self.selection_grn(flatten_inputs, context, training=training) # (batch_size, window, num_inputs)
        weights = self.softmax(weights)  # (batch_size, window, num_inputs)
        weights = ops.expand_dims(weights, axis=-1)  # (batch_size, window, num_inputs, 1)

        # Process each input with it's corresponding GRN
        grn_outputs = [grn(inputs[..., i, :], context, training=training) for i, grn in enumerate(self.grns)]  # (batch_size, window, hidden_units) * num_inputs
        transformed = ops.stack(grn_outputs, axis=-2)  # (batch_size, window, num_inputs, hidden_units)

        # Multiply weights and sum
        combined = ops.multiply(transformed, weights)  # (batch_size, window, num_inputs, hidden_units)
        output = ops.sum(combined, axis=-2)  # (batch_size, window, hidden_units)

        return output





class TFT(keras.Model):
    
    def __init__(self, hidden_units, output_size, dropout_rate=None, **kwargs):
        super().__init__(**kwargs)

        self.hidden_units = hidden_units
        self.output_size = output_size
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        #self.gnu = GRN(self.hidden_units, output_size=1)
        self.vsn = VariableSelectionNetwork(self.hidden_units, dropout_rate=self.dropout_rate, time_distributed=True)


        self.output_layer = layers.TimeDistributed(layers.Dense(self.output_size))
        
    def call(self, inputs):

        x = self.vsn(inputs)

        #x = self.output_layer(x)

        return x
    

if __name__ == "__main__":
    hidden_units = 16
    timesteps = 10
    num_inputs = 16
    embedding_dim = 3
    batch_size = 2


    inputs = ops.ones((batch_size, timesteps, num_inputs, embedding_dim))
    inputs2 = keras.Input(inputs.shape)

    model = TFT(hidden_units, output_size=1)
    model.compile()
    model.build(input_shape=inputs.shape)
    output = model(inputs, training=True)

    print("Input shape:", inputs.shape)
    print("Output shape:", output.shape)
