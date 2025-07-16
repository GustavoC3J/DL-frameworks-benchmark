
import torch
import torch.nn as nn
import torch.nn.functional as F

from runners.model_builder.models.torch.tft.gated_residual_network import GatedResidualNetwork
from runners.model_builder.models.torch.tft.glu import GLU
from runners.model_builder.models.torch.tft.input_embedding import InputEmbedding
from runners.model_builder.models.torch.tft.static_covariate_encoder import StaticCovariateEncoder
from runners.model_builder.models.torch.tft.variable_selection_network import VariableSelectionNetwork


class TFT(nn.Module):
    def __init__(
        self,
        hidden_units,
        output_size,
        num_attention_heads,
        historical_window,
        prediction_window,
        observed_idx, # observed (target/s)
        static_idx=[], # static
        known_idx=[], # known
        unknown_idx=[], # unknown
        categorical_idx=[],
        categorical_counts=[], # Number of categories for each categorical variable.
        dropout_rate=0.0,
    ):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.historical_window = historical_window
        self.prediction_window = prediction_window

        self.observed_idx = observed_idx
        self.static_idx = static_idx
        self.known_idx = known_idx
        self.unknown_idx = unknown_idx
        self.categorical_idx = categorical_idx

        self.num_inputs = len(self.observed_idx) + len(self.static_idx) + len(self.known_idx) + len(self.unknown_idx)
        self.num_static_inputs = len(self.static_idx)
        self.num_historical_inputs = self.num_inputs - self.num_static_inputs
        self.num_future_inputs = len(self.known_idx)

        self.output_size = output_size

        if categorical_idx and not categorical_counts:
            raise ValueError('"categorical_counts" is required when "categorical_indexes" is provided.')

        # Embeddings
        self.embedding_layer = InputEmbedding(
            num_inputs=self.num_inputs,
            embedding_dim=hidden_units,
            historical_window=historical_window,
            prediction_window=prediction_window,
            static_idx=static_idx,
            observed_idx=observed_idx,
            known_idx=known_idx,
            unknown_idx=unknown_idx,
            categorical_idx=categorical_idx,
            categorical_counts=categorical_counts
        )

        # Static
        if self.num_static_inputs == 0:
            self.static_encoder = None
            self.static_vsn = None
        else:
            self.static_vsn = VariableSelectionNetwork(
                hidden_units=hidden_units,
                num_inputs=self.num_static_inputs,
                dropout_rate=dropout_rate,
                time_distributed=False
            )

            self.static_encoder = StaticCovariateEncoder(
                hidden_dim=hidden_units,
                dropout_rate=dropout_rate
            )

        # historical variable selection network
        self.historical_vsn = VariableSelectionNetwork(
            hidden_units=hidden_units,
            num_inputs=self.num_historical_inputs,
            dropout_rate=dropout_rate,
            time_distributed=True
        )

        # Future variable selection network
        if self.num_future_inputs == 0:
            self.future_vsn = None
        else:
            self.future_vsn = VariableSelectionNetwork(
                hidden_units=hidden_units,
                num_inputs=self.num_future_inputs,
                dropout_rate=dropout_rate,
                time_distributed=True
            )

        # LSTM history and future
        self.encoder_lstm = nn.LSTM(
            hidden_units,
            hidden_units,
            batch_first=True
        )

        self.decoder_lstm = nn.LSTM(
            hidden_units,
            hidden_units,
            batch_first=True
        )

        self.glu_lstm = GLU(
            hidden_units,
            hidden_units,
            dropout_rate=dropout_rate,
            time_distributed=True
        )

        self.layer_norm_lstm = nn.LayerNorm(hidden_units)

        # Context enrichment
        self.grn_enrichment = GatedResidualNetwork(
            input_dim=hidden_units,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            time_distributed=True
        )

        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_units,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        self.glu_multihead = GLU(
            hidden_units,
            hidden_units,
            dropout_rate=dropout_rate,
            time_distributed=True
        )

        self.layer_norm_multihead = nn.LayerNorm(hidden_units)

        self.grn_multihead = GatedResidualNetwork(
            input_dim=hidden_units,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            time_distributed=True
        )

        # Output layer
        self.glu_output = GLU(
            hidden_units,
            hidden_units,
            time_distributed=True
        )

        self.layer_norm_output = nn.LayerNorm(hidden_units)

        self.output_layer = nn.Linear(hidden_units, output_size)

    def forward(self, inputs):
        """
        inputs: Tensor of shape (batch_size, window, num_inputs)
        """

        batch_size = inputs.shape[0]

        # Apply the embedding layer and split inputs into static, historical, and future
        static_input, historical_input, future_input = self.embedding_layer(inputs)

        # 1. Static covariate encoding
        if self.num_static_inputs and static_input is not None:
            static_selected = self.static_vsn(static_input)
            static_context = self.static_encoder(static_selected)

            context_var_sel, context_enrichment, (context_state_h, context_state_c) = static_context
        else:
            # If there are no static inputs, set to None and zeros for context state
            context_var_sel = context_enrichment = None
            context_state_h = context_state_c = torch.zeros((batch_size, self.hidden_units), device=inputs.device)

        # 2. Variable selection
        historical_features = self.historical_vsn(historical_input, context=context_var_sel)

        if self.num_future_inputs and future_input is not None:
            future_features = self.future_vsn(future_input, context=context_var_sel)
        else:
            future_features = torch.zeros((batch_size, self.prediction_window, self.hidden_units), device=inputs.device)

        # 3. LSTM encoder/decoder
        encoder_output, (state_h, state_c) = self.encoder_lstm(
            historical_features,
            (context_state_h.unsqueeze(0), context_state_c.unsqueeze(0)) # Expects (1, batch_size, hidden_units)
        )

        # Use the last state of the encoder as the initial state for the encoder
        decoder_output, _ = self.decoder_lstm(
            future_features, (state_h, state_c)
        )

        lstm_layer = torch.cat([encoder_output, decoder_output], dim=1)

        # Apply gated skip connection
        input_embeddings = torch.cat([historical_features, future_features], dim=1)

        lstm_layer = self.glu_lstm(lstm_layer)
        temporal_feature_layer = lstm_layer + input_embeddings
        temporal_feature_layer = self.layer_norm_lstm(temporal_feature_layer)

        # 4. Static enrichment
        enriched = self.grn_enrichment(
            temporal_feature_layer,
            context=context_enrichment
        )

        # 5. Self-attention => same tensor for query, key, and value
        attn_output, _ = self.self_attention(enriched, enriched, enriched)

        x = self.glu_multihead(attn_output)
        x = x + enriched
        x = self.layer_norm_multihead(x)

        decoder = self.grn_multihead(x)

        # 6. Final skip connection and output layer
        decoder = self.glu_output(decoder)
        x = decoder + temporal_feature_layer
        transformer_layer = self.layer_norm_output(x)

        output = self.output_layer(transformer_layer)

        # Keep only the prediction window
        output = output[:, -self.prediction_window:, :]

        return output
