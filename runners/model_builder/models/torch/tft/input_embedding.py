
import torch
import torch.nn as nn
import torch.nn.functional as F

class InputEmbedding(nn.Module):
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
            categorical_idx=[],
            categorical_counts=[], # Number of categories for each categorical variable.
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

        self.embedding_layers = nn.ModuleList()

        for i in range(self.num_inputs):
            if i in categorical_idx:
                # If it's a categorical variable, use an embedding layer
                count = categorical_counts[categorical_idx.index(i)]

                if count <= 0:
                    raise ValueError(f'Invalid count {count} for categorical index {i}. Must be greater than 0.')
                
                self.embedding_layers.append(nn.Embedding(count, embedding_dim))
            
            else:
                self.embedding_layers.append(nn.Linear(1, embedding_dim))

    def forward(self, inputs: torch.Tensor):
        """
        input: Tensor of shape (batch_size, window, num_inputs)

        Static: Static features
        Historical: Observed, known and unknown features
        Future: Known features
        """
        
        input_shape = inputs.shape
        total_window = self.historical_window + self.prediction_window

        # If inputs' window is longer than needed, slice it
        if input_shape[1] > total_window:
            inputs = inputs[:, -total_window:, :]


        static_inputs = []
        historical_inputs = []
        future_inputs = []
        
        for i in range(self.num_inputs):
            # Apply the appropriate embedding layer for each input
            output = self.embedding_layers[i](inputs[:, :, i:i+1])

            if output.ndim == 3:
                output = output.unsqueeze(2)
            
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
            static_inputs = torch.cat(static_inputs, dim=-2)  # (batch, window, num_static, embedding_dim)
            static_inputs = static_inputs[:, 0, :, :]  # (batch, num_static, embedding_dim)
        else:
            static_inputs = None


        historical_inputs = torch.cat(historical_inputs, dim=-2)  # (batch, window, num_hist, embedding_dim)
        historical_inputs = historical_inputs[:, :self.historical_window, :, :]  # (batch, historical_window, num_hist, embedding_dim)

        if future_inputs:
            future_inputs = torch.cat(future_inputs, dim=-2)  # (batch, window, num_known, embedding_dim)
            future_inputs = future_inputs[:, self.historical_window:, :, :]  # (batch, prediction_window, num_known, embedding_dim)
        else:
            future_inputs = None

        return static_inputs, historical_inputs, future_inputs
