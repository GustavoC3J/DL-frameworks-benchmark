
import torch.nn as nn
from runners.model_builder.models.torch.tft.gated_residual_network import GatedResidualNetwork

class StaticCovariateEncoder(nn.Module):
    """StaticCovariateEncoder"""
    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

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

    def forward(self, inputs):
        """
        inputs: Tensor of shape (batch_size, hidden_size)
        """
        
        c_variable_selection = self.context_grn_variable_selection(inputs)
        c_enrichment = self.context_grn_enrichment(inputs)
        c_state_h = self.context_grn_state_h(inputs)
        c_state_c = self.context_grn_state_c(inputs)

        return (
            c_variable_selection, # Context variable selection 
            c_enrichment, # Context enrichment
            (c_state_h, c_state_c) # Context state (h, c)
        )
