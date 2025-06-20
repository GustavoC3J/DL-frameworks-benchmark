
import keras
from keras import layers, ops

from runners.model_builder.models.keras.tft.gated_residual_network import GatedResidualNetwork

class StaticCovariateEncoder(layers.Layer):
    """StaticCovariateEncoder"""
    def __init__(self, hidden_dim, num_static_vars, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim

        # GRN for each static variable (applied per variable)
        self.static_variable_grn = GatedResidualNetwork(
            hidden_units=hidden_dim,
            dropout_rate=dropout_rate,
            time_distributed=False
        )

        # GRN for combining (output size = num_static_vars)
        self.static_combine_grn = GatedResidualNetwork(
            hidden_units=hidden_dim,
            output_size=num_static_vars,
            dropout_rate=dropout_rate,
            time_distributed=False
        )

        # Context GRNs
        self.context_grn_variable_selection = GatedResidualNetwork(
            hidden_units=hidden_dim,
            dropout_rate=dropout_rate,
            time_distributed=False
        )
        self.context_grn_enrichment = GatedResidualNetwork(
            hidden_units=hidden_dim,
            dropout_rate=dropout_rate,
            time_distributed=False
        )
        self.context_grn_state_h = GatedResidualNetwork(
            hidden_units=hidden_dim,
            dropout_rate=dropout_rate,
            time_distributed=False
        )
        self.context_grn_state_c = GatedResidualNetwork(
            hidden_units=hidden_dim,
            dropout_rate=dropout_rate,
            time_distributed=False
        )

    def call(self, static_embeddings):
        """
        static_embeddings: Tensor of shape (batch_size, num_static_vars, embedding_dim)

        """
        batch_size, num_static_vars, _ = static_embeddings.shape

        # Apply GRN to each static variable separately
        grn_outputs = []
        for i in range(num_static_vars):
            grn_out = self.static_variable_grn(static_embeddings[:, i:i+1, :]) # (batch_size, 1, hidden_size)
            grn_outputs.append(grn_out)
        grn_output = ops.concatenate(grn_outputs, axis=1) # (batch_size, num_static_vars, hidden_size)

        # Flatten static embeddings for combine GRN
        flatten = ops.reshape(static_embeddings, (batch_size, -1)) # (batch_size, num_static_vars * embedding_dim)
        
        combine_weights = self.static_combine_grn(flatten) # (batch_size, num_static_vars)
        variable_weights = ops.softmax(combine_weights, axis=1) # (batch_size, num_static_vars)
        variable_weights = ops.expand_dims(variable_weights, axis=-1) # (batch_size, num_static_vars, 1)

        # Weighted sum
        static_context_vector = ops.sum(variable_weights * grn_output, axis=1)  # (batch_size, hidden_size)

        # Contexts
        c_variable_selection = self.context_grn_variable_selection(static_context_vector)
        c_enrichment = self.context_grn_enrichment(static_context_vector)
        c_state_h = self.context_grn_state_h(static_context_vector)
        c_state_c = self.context_grn_state_c(static_context_vector)

        return {
            "context_variable_selection": c_variable_selection,
            "context_enrichment": c_enrichment,
            "context_state": (c_state_h, c_state_c)
        }