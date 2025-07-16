
from keras import layers, ops

from runners.model_builder.models.keras.tft.gated_residual_network import GatedResidualNetwork


class VariableSelectionNetwork(layers.Layer):
    """Variable Selection Network (VSN)"""

    def __init__(self, hidden_units, num_inputs, dropout_rate=None, time_distributed=True, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.time_distributed = time_distributed

        
        self.grns = [
            GatedResidualNetwork(
                self.hidden_units,
                dropout_rate=self.dropout_rate,
                time_distributed=self.time_distributed
            )
            for _ in range(num_inputs)
        ]
        
        self.softmax = layers.Softmax()
        self.flatten = layers.Reshape( 
            (-1, num_inputs * hidden_units) if self.time_distributed else \
            (num_inputs * hidden_units,)
        )
        

        self.selection_grn = GatedResidualNetwork(
            self.hidden_units,
            output_size=num_inputs,  # One for each feature
            dropout_rate=self.dropout_rate,
            time_distributed=self.time_distributed
        )
        
    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, context=None, training=None):
        # inputs: (batch_size, num_inputs, hidden_units) or (batch_size, window, num_inputs, hidden_units)

        input_shape = ops.shape(inputs)

        # Variable selection weights
        flatten_inputs = self.flatten(inputs)  # (batch_size, <window,> num_inputs * hidden_units)
        weights = self.selection_grn(flatten_inputs, context, training=training) # (batch_size, <window,> num_inputs)
        weights = self.softmax(weights)  # (batch_size, <window,> num_inputs)
        weights = ops.expand_dims(weights, axis=-1)  # (batch_size, <window,> num_inputs, 1)

        # Process each input with it's corresponding GRN
        grn_outputs = []
        for i, grn in enumerate(self.grns):
            input_slice = ops.slice(
                inputs,
                start_indices = [0, 0, i, 0]                                    if self.time_distributed else [0, i, 0],
                shape = [input_shape[0], input_shape[1], 1, self.hidden_units]  if self.time_distributed else [input_shape[0], 1, self.hidden_units]
            )

            grn_outputs.append( grn(input_slice, context, training=training) ) # (batch_size, <window,> hidden_units) * num_inputs

        transformed = ops.concatenate(grn_outputs, axis=-2)  # (batch_size, <window,> num_inputs, hidden_units)

        # Multiply weights and sum
        combined = ops.multiply(transformed, weights)  # (batch_size, <window,> num_inputs, hidden_units)
        output = ops.sum(combined, axis=-2)  # (batch_size, <window,> hidden_units)

        return output
    
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.hidden_units) if self.time_distributed else (input_shape[0], self.hidden_units)
