
from dataclasses import field
from typing import List, Optional

import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype
from jax import Array

from runners.model_builder.models.flax.lstm import LSTM
from runners.model_builder.models.flax.tft.gated_residual_network import \
    GatedResidualNetwork
from runners.model_builder.models.flax.tft.glu import GLU
from runners.model_builder.models.flax.tft.input_embedding import \
    InputEmbedding
from runners.model_builder.models.flax.tft.static_covariate_encoder import \
    StaticCovariateEncoder
from runners.model_builder.models.flax.tft.variable_selection_network import \
    VariableSelectionNetwork


class TFT(nn.Module):
    hidden_units: int
    output_size: int
    num_attention_heads: int
    historical_window: int
    prediction_window: int
    observed_idx: List[int]
    static_idx: List[int] = field(default_factory=list)
    known_idx: List[int] = field(default_factory=list)
    unknown_idx: List[int] = field(default_factory=list)
    categorical_idx: List[int] = field(default_factory=list)
    categorical_counts: List[int] = field(default_factory=list)
    dropout_rate: Optional[float] = None
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: Array, *, training: bool = False):
        """
        inputs: Tensor of shape (batch_size, window, num_inputs)
        """

        num_inputs = len(self.observed_idx) + len(self.static_idx) + len(self.known_idx) + len(self.unknown_idx)
        num_static_inputs = len(self.static_idx)
        num_historical_inputs = num_inputs - num_static_inputs
        num_future_inputs = len(self.known_idx)

        batch_size = inputs.shape[0]

        # Embeddings
        embedding_layer = InputEmbedding(
            num_inputs=num_inputs,
            embedding_dim=self.hidden_units,
            historical_window=self.historical_window,
            prediction_window=self.prediction_window,
            static_idx=self.static_idx,
            observed_idx=self.observed_idx,
            known_idx=self.known_idx,
            unknown_idx=self.unknown_idx,
            categorical_idx=self.categorical_idx,
            categorical_counts=self.categorical_counts,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        static_input, historical_input, future_input = embedding_layer(inputs, training=training)

        # 1. Static covariate encoding
        if num_static_inputs and static_input is not None:
            static_vsn = VariableSelectionNetwork(
                hidden_units=self.hidden_units,
                num_inputs=num_static_inputs,
                dropout_rate=self.dropout_rate,
                time_distributed=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
            static_selected = static_vsn(static_input, training=training)

            static_encoder = StaticCovariateEncoder(
                hidden_dim=self.hidden_units,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
            context_var_sel, context_enrichment, (context_state_h, context_state_c) = static_encoder(static_selected, training=training)
        else:
            context_var_sel = context_enrichment = None
            context_state_h = context_state_c = jnp.zeros((batch_size, self.hidden_units), dtype=self.dtype or jnp.float32)

        # 2. Variable selection
        historical_vsn = VariableSelectionNetwork(
            hidden_units=self.hidden_units,
            num_inputs=num_historical_inputs,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        historical_features = historical_vsn(historical_input, context=context_var_sel, training=training)

        if num_future_inputs and future_input is not None:
            future_vsn = VariableSelectionNetwork(
                hidden_units=self.hidden_units,
                num_inputs=num_future_inputs,
                dropout_rate=self.dropout_rate,
                time_distributed=True,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
            future_features = future_vsn(future_input, context=context_var_sel, training=training)
        else:
            future_features = jnp.zeros((batch_size, self.prediction_window, self.hidden_units), dtype=self.dtype or jnp.float32)

        # 3. LSTM encoder/decoder
        encoder_lstm = LSTM(
            features=self.hidden_units,
            return_sequences=True,
            return_state=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        encoder_output, state_h, state_c = encoder_lstm(
            historical_features,
            initial_state=(context_state_h, context_state_c),
            training=training,
        )

        # Use the last state of the encoder as the initial state for the encoder
        decoder_lstm = LSTM(
            features=self.hidden_units,
            return_sequences=True,
            return_state=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        decoder_output = decoder_lstm(
            future_features,
            initial_state=(state_h, state_c),
            training=training,
        )

        lstm_layer = jnp.concatenate([encoder_output, decoder_output], axis=1)

        # Apply gated skip connection
        input_embeddings = jnp.concatenate([historical_features, future_features], axis=1)

        glu_lstm = GLU(
            hidden_units=self.hidden_units,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        lstm_layer = glu_lstm(lstm_layer, training=training)
        temporal_feature_layer = lstm_layer + input_embeddings
        temporal_feature_layer = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)(temporal_feature_layer)

        # 4. Static enrichment
        grn_enrichment = GatedResidualNetwork(
            hidden_units=self.hidden_units,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        enriched = grn_enrichment(temporal_feature_layer, context=context_enrichment, training=training)

        # 5. Self-attention
        attn_output = nn.SelfAttention(
            num_heads=self.num_attention_heads,
            qkv_features=self.hidden_units,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(enriched, deterministic=not training)

        glu_multihead = GLU(
            hidden_units=self.hidden_units,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        x = glu_multihead(attn_output, training=training)
        x = x + enriched
        x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)

        grn_multihead = GatedResidualNetwork(
            hidden_units=self.hidden_units,
            dropout_rate=self.dropout_rate,
            time_distributed=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        decoder = grn_multihead(x, training=training)

        # 6. Final skip connection and output layer
        glu_output = GLU(
            hidden_units=self.hidden_units,
            time_distributed=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        decoder = glu_output(decoder, training=training)
        x = decoder + temporal_feature_layer
        transformer_layer = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)

        # Output layer
        output = nn.Dense(
            features=self.output_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(transformer_layer)

        # Keep only the prediction window
        output = output[:, -self.prediction_window:, :]

        return output
