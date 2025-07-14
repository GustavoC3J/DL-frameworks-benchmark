
import torch.nn as nn
import torch.nn.functional as F
from glu import GLU

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN)"""

    def __init__(self, hidden_units, output_size=None, dropout_rate=None, time_distributed=True):
        super().__init__()
        self.hidden_units = hidden_units
        self.output_size = output_size if output_size is not None else hidden_units
        self.dropout_rate = dropout_rate
        self.time_distributed = time_distributed

        # Projection layer for residual connection (if needed)
        self.projection = nn.Linear(hidden_units, self.output_size)

        # Feed-forward layers
        self.dense1 = nn.Linear(hidden_units, hidden_units)
        self.context_dense = nn.Linear(hidden_units, hidden_units, bias=False)
        self.dense2 = nn.Linear(hidden_units, hidden_units)

        # Gating layer
        self.glu = GLU(self.output_size, dropout_rate, time_distributed)

        # Normalization
        self.layer_norm = nn.LayerNorm(self.output_size)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else nn.Identity()

    def forward(self, inputs, context=None):
        """
        Args:
            inputs: shape [batch, seq, features] or [batch, features]
            context: shape [batch, features] (optional)
        """
        
        # Residual connection
        # If input dimension doesn't match output's, apply a projection
        residual = self.projection(inputs) if inputs.shape[-1] != self.output_size else inputs

        # First linear and add context (if available)
        x = self.dense1(inputs)

        if context is not None:
            # If it's time series, add temporal dimension to context
            if self.time_distributed:
                context = context.unsqueeze(1)
                
            context_transformed = self.context_dense(context)
            x = x + context_transformed

        # Activation, second linear, and dropout
        x = F.elu(x)
        x = self.dense2(x)
        x = self.dropout(x)

        # Gating (GLU)
        x = self.glu(x)

        # Residual connection + normalization
        x = x + residual
        x = self.layer_norm(x)

        return x
