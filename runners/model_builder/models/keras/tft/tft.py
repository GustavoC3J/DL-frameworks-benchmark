

from runners.model_builder.models.keras.tft.input_embedding import InputEmbedding
from runners.model_builder.models.keras.tft.gated_residual_network import GatedResidualNetwork
from runners.model_builder.models.keras.tft.glu import GLU
from runners.model_builder.models.keras.tft.static_covariate_encoder import StaticCovariateEncoder
from runners.model_builder.models.keras.tft.variable_selection_network import VariableSelectionNetwork

import keras
from keras import layers, ops

class TFT(keras.Model):
    def __init__(
        self,
        hidden_units,
        output_size,
        num_attention_heads,
        historical_window,
        prediction_window,
        observed_idx: list, # observed (target/s)
        static_idx: list=[], # static
        known_idx: list=[], # known
        unknown_idx: list=[], # unknown
        categorical_idx: list=[],
        categorical_counts: list=[], # Number of categories for each categorical variable.
        dropout_rate=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
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
            # Variable selection network
            self.static_vsn = VariableSelectionNetwork(
                hidden_units=hidden_units,
                num_inputs=self.num_static_inputs,
                dropout_rate=dropout_rate,
                time_distributed=False
            )

            # Static covariate encoder
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
        self.encoder_lstm = layers.LSTM(
            hidden_units,
            return_sequences=True,
            return_state=True
        )

        self.decoder_lstm = layers.LSTM(
            hidden_units,
            return_sequences=True,
            return_state=False
        )

        self.glu_lstm = GLU(
            hidden_units,
            dropout_rate=dropout_rate,
            time_distributed=True
        )

        self.layer_norm_lstm = layers.LayerNormalization()

        # Context enrichment
        self.grn_enrichment = GatedResidualNetwork(
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            time_distributed=True
        )

        # Multi-head self-attention
        self.self_attention = layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=hidden_units // num_attention_heads,
            dropout=dropout_rate
        )

        self.glu_multihead = GLU(
            hidden_units,
            dropout_rate=dropout_rate,
            time_distributed=True
        )

        self.layer_norm_multihead = layers.LayerNormalization()

        
        self.grn_multihead = GatedResidualNetwork(
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            time_distributed=True
        )

        # Output layer
        self.glu_output = GLU(
            hidden_units,
            time_distributed=True
        )

        self.layer_norm_output = layers.LayerNormalization()

        self.output_layer = layers.TimeDistributed(layers.Dense(output_size))


    def build(self, input_shape):
        super().build(input_shape)


    def call(self, inputs, training=None):
        """
        inputs: Tensor of shape (batch_size, window, num_inputs)
        """

        batch_size = ops.shape(historical_input)[0]

        # Apply the embedding layer and split inputs into static, historical, and future
        static_input, historical_input, future_input = self.embedding_layer(inputs)


        # 1. Static covariate encoding
        if self.num_static_inputs is not None and static_input is not None:
            static_selected = self.static_vsn(static_input)
            static_context = self.static_encoder(static_selected)

            context_var_sel, context_enrichment, (context_state_h, context_state_c) = static_context
        else:
            # If there are no static inputs, set to None and zeros for context state
            context_var_sel = context_enrichment = None
            context_state_h = context_state_c = ops.zeros((batch_size, self.hidden_units))


        # 2. Variable selection
        historical_features = self.historical_vsn(historical_input, context=context_var_sel)

        if self.num_future_inputs is not None and future_input is not None:
            future_features = self.future_vsn(future_input, context=context_var_sel)
        else:
            future_features = ops.zeros((batch_size, self.prediction_window, self.hidden_units))


        # 3. LSTM encoder/decoder
        encoder_output, state_h, state_c = self.encoder_lstm(
            historical_features,
            initial_state=[context_state_h, context_state_c]
        )

        # Use the last state of the encoder as the initial state for the encoder
        decoder_output = self.decoder_lstm(
            future_features,
            initial_state=[state_h, state_c]
        )

        lstm_layer = ops.concatenate([encoder_output, decoder_output], axis=1)

        # Apply gated skip connection
        input_embeddings = ops.concatenate([historical_features, future_features], axis=1)

        lstm_layer = self.glu_lstm(lstm_layer)
        temporal_feature_layer = ops.add(lstm_layer, input_embeddings)
        temporal_feature_layer = self.layer_norm_lstm(temporal_feature_layer)

        # 4. Static enrichment
        enriched = self.grn_enrichment(
            temporal_feature_layer,
            context=context_enrichment
        )

        # 5. Self-attention => same tensor for query, key, and value
        attn_output = self.self_attention(enriched, enriched, enriched)

        x = self.glu_multihead(attn_output)
        x = ops.add(x, enriched)
        x = self.layer_norm_multihead(x)

        decoder = self.grn_multihead(x)

        # 6. Final skip connection and output layer
        decoder = self.glu_output(decoder)
        x = ops.add(decoder, temporal_feature_layer)
        transformer_layer = self.layer_norm_output(x)

        output = self.output_layer(transformer_layer)

        # Keep only the prediction window
        output = output[:, -self.prediction_window:, :]

        return output