
import keras
from keras import layers, ops

class InputEmbedding(layers.Layer):
    """Embedding layer for TFT model."""
    
    def __init__(
            self,
            num_inputs,
            embedding_dim,
            historical_window,
            prediction_window,
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
        self.prediction_window = prediction_window

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

        Static: Static features
        Historical: Observed, known and unknown features
        Future: Known features
        """

        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        window_size = self.historical_window + self.prediction_window

        # If inputs' window is longer than needed, slice it
        if input_shape[1] > window_size:
            inputs = ops.slice(inputs, [0, input_shape[1] - window_size, 0], [batch_size, window_size, input_shape[2]])


        static_inputs = []
        historical_inputs = []
        future_inputs = []
        
        for i in range(self.num_inputs):
            # Apply the appropriate embedding layer for each input
            output = self.embedding_layers[i]( ops.slice(inputs, [0, 0, i], [batch_size, window_size, 1]) ) # inputs[:, :, i:i+1]
            
            if output.ndim == 3:
                output = ops.expand_dims(output, axis=2)

            # Append to the corresponding feature list
            if i in self.static_idx:
                static_inputs.append(output)

            elif i in self.known_idx:
                historical_inputs.append(output)
                future_inputs.append(output)

            else: # observed, unknown
                historical_inputs.append(output)

        
        # Join each list into a tensor

        if static_inputs:
            # Keep only first element (repeat it later if needed)
            static_inputs = ops.slice( # (batch_size, 1, num_static, embedding_dim)
                ops.concatenate(static_inputs, axis=-2),
                start_indices=[0, 0, 0, 0],
                shape=[batch_size, 1, len(self.static_idx), self.embedding_dim]
            )
            static_inputs = ops.squeeze(static_inputs, axis=1) # (batch_size, num_static, embedding_dim)
        else:
            static_inputs = None


        historical_inputs = ops.slice(  # (batch_size, window, num observed + unknown + known, embedding_dim)
            ops.concatenate(historical_inputs, axis=-2),
            start_indices=[0, 0, 0, 0],
            shape=[batch_size, self.historical_window, len(historical_inputs), self.embedding_dim]
        )

        if future_inputs:
            future_inputs = ops.slice(  # (batch_size, prediction_window, num known, embedding_dim)
                ops.concatenate(future_inputs, axis=-2),
                start_indices=[0, self.historical_window, 0, 0],
                shape=[batch_size, self.prediction_window, len(self.known_idx), self.embedding_dim]
            )
        else:
            future_inputs = None


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