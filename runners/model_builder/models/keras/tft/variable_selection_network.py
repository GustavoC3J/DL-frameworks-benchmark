
from keras import layers, ops

from runners.model_builder.models.keras.tft.gated_residual_network import GatedResidualNetwork


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
            GatedResidualNetwork(
                self.hidden_units,
                dropout_rate=self.dropout_rate,
                time_distributed=self.time_distributed
            )
            for _ in range(num_inputs)
        ]
        
        self.softmax = layers.Softmax()
        self.flatten = layers.Reshape( (input_shape[1], num_inputs * embedding_dim) if self.time_distributed else \
                                       (num_inputs * embedding_dim) )

        self.selection_grn = GatedResidualNetwork(
            self.hidden_units,
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
