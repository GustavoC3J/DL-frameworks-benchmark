
import keras
from keras import layers, ops

class InputEmbedding(layers.Layer):
    """Embedding layer for TFT model."""
    
    def __init__(
            self,
            num_inputs,
            embedding_dim,
            historical_window,
            static_idx, # static
            observed_idx, # observed (target(s))
            known_idx, # known
            unknown_idx, # unknown
            categorical_idx: list=[],
            categorical_counts: list=[], # Number of categories for each categorical variable.
        
        ):
        super().__init__()

        self.num_inputs = num_inputs
        self.embedding_dim = embedding_dim
        self.historical_window = historical_window

        self.static_idx = static_idx
        self.observed_idx = observed_idx
        self.known_idx = known_idx
        self.unknown_idx = unknown_idx

        self.embedding_layers = []

        for i in range(self.num_inputs):
            if i in categorical_idx:
                # If it's a categorical variable, use an embedding layer
                count = categorical_counts[categorical_idx.index(i)]

                if count <= 0:
                    raise ValueError(f'Invalid count {count} for categorical index {i}. Must be greater than 0.')
                
                self.embedding_layers.append(layers.Embedding(input_dim=count, output_dim=embedding_dim, name=f'cat_embedding_{i}'))

            else:
                # If it's not a categorical variable, use a Dense layer to project the input to the embedding dimension
                self.embedding_layers.append( layers.TimeDistributed( layers.Dense(embedding_dim, name=f'dense_embedding_{i}') ) )


    def build(self, input_shape):
        super().build(input_shape)


    def call(self, inputs):
        """
        input: Tensor of shape (batch_size, window, num_inputs)
        """
        # Apply the appropriate embedding layer for each input
        embedded_inputs = []
        for i in range(self.num_inputs):
            output = self.embedding_layers[i](inputs[:, :, i:i+1])
            
            if output.ndim == 3:
                output = ops.expand_dims(output, axis=2)

            embedded_inputs.append(output)  # list of [i](batch_size, window, 1, embedding_dim)

        # Static inputs: Keep only first element (repeat it later if needed)
        if self.static_idx:
            static_inputs = [embedded_inputs[i][:, 0, :, :] for i in self.static_idx]
            static_inputs = ops.concatenate(static_inputs, axis=1)  # (batch_size, num_static, embedding_dim)
        else:
            static_inputs = None

        # Observed, known and unknown inputs
        observed_inputs = [embedded_inputs[i] for i in self.observed_idx]
        observed_inputs = ops.concatenate(observed_inputs, axis=-2) # (batch_size, window, num_observed, embedding_dim)

        known_inputs = [embedded_inputs[i] for i in self.known_idx]
        known_inputs = ops.concatenate(known_inputs, axis=-2) if known_inputs else None # (batch_size, window, num_known, embedding_dim)
        
        unknown_inputs = [embedded_inputs[i] for i in self.unknown_idx]
        unknown_inputs = ops.concatenate(unknown_inputs, axis=-2) if unknown_inputs else None # (batch_size, window, num_unkown, embedding_dim)


        # Historical and future inputs
        historical_inputs = []

        if (unknown_inputs is not None):
            historical_inputs.append(unknown_inputs[:, :self.historical_window, :, :])

        if (known_inputs is not None):
            historical_inputs.append(known_inputs[:, :self.historical_window, :, :])

        historical_inputs.append(observed_inputs[:, :self.historical_window, :, :])

        historical_inputs = ops.concatenate(historical_inputs, axis=-2) # (batch_size, historical_window, num_observed + num_known + num_unkown, embedding_dim)
        

        future_inputs = known_inputs[:, self.historical_window:, :, :] if known_inputs is not None else None # (batch_size, prediction_window, num_unkown, embedding_dim)

        return static_inputs, historical_inputs, future_inputs
    

    def compute_output_shape(self, input_shape):
        """
        output: Tensors of shape 
            (batch_size, num_static, embedding_dim),
            (batch_size, historical_window, num_observed + num_known + num_unkown, embedding_dim),
            (batch_size, prediction_window, num_unkown, embedding_dim)
        """
        return (input_shape[0], len(self.static_idx), self.embedding_dim), \
               (input_shape[0], self.historical_window, len(self.observed_idx) + len(self.known_idx) + len(self.unknown_idx), self.embedding_dim), \
               (input_shape[0], input_shape[1] - self.historical_window, len(self.known_idx), self.embedding_dim)