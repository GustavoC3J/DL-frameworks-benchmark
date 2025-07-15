
import torch
import torch.nn as nn
import torch.nn.functional as F
from runners.model_builder.models.torch.tft.gated_residual_network import GatedResidualNetwork

class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN)"""

    def __init__(self, hidden_units, num_inputs, dropout_rate=None, time_distributed=True):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.time_distributed = time_distributed

        # Selection GRN to get the weights for variable selection
        self.selection_grn = GatedResidualNetwork(
            input_dim=num_inputs * hidden_units,
            hidden_units=hidden_units,
            output_size=num_inputs,  # One for each feature
            dropout_rate=dropout_rate,
            time_distributed=time_distributed
        )
        
        self.softmax = nn.Softmax(dim=-1)

        # GRN for each input variable
        self.grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=hidden_units,
                hidden_units=hidden_units,
                dropout_rate=dropout_rate,
                time_distributed=time_distributed
            )
            for _ in range(num_inputs)
        ])

        
    def forward(self, inputs: torch.Tensor, context=None):
        # inputs: (batch_size, num_inputs, hidden_units) or (batch_size, window, num_inputs, hidden_units)

        # Variable selection weights
        flatten_inputs = inputs.flatten(start_dim=-2)  # (batch_size, <window,> num_inputs * hidden_units)
        weights = self.selection_grn(flatten_inputs, context) # (batch_size, <window,> num_inputs)
        weights = self.softmax(weights)  # (batch_size, <window,> num_inputs)
        weights = weights.unsqueeze(-1)  # (batch_size, <window,> num_inputs, 1)

        # Process each input with it's corresponding GRN
        grn_outputs = []
        for i, grn in enumerate(self.grns):
            input_slice = inputs[:, :, i:i+1, :] if self.time_distributed else inputs[:, i:i+1, :]
            grn_outputs.append(grn(input_slice, context)) # (batch_size, <window,> hidden_units) * num_inputs

        transformed = torch.cat(grn_outputs, dim=-2)  # (batch_size, <window,> num_inputs, hidden_units)

        # Multiply weights and sum
        combined = transformed * weights  # (batch_size, <window,> num_inputs, hidden_units)
        output = combined.sum(dim=-2)  # (batch_size, <window,> hidden_units)

        return output
