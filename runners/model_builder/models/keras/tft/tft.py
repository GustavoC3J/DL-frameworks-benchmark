

from runners.model_builder.models.keras.tft.gated_residual_network import GatedResidualNetwork
from runners.model_builder.models.keras.tft.glu import GLU
from runners.model_builder.models.keras.tft.static_covariate_encoder import StaticCovariateEncoder
from runners.model_builder.models.keras.tft.variable_selection_network import VariableSelectionNetwork

import keras
from keras import layers, ops

# TODO: embeddings?
class TFT(keras.Model):
    def __init__(
        self,
        hidden_units,
        num_historic_inputs,
        embedding_dim,
        output_size,
        num_attention_heads,
        dropout_rate=0.0,
        num_static_inputs=None, # If it's None, static inputs will not be used.
        num_future_inputs=None, # If it's None, future inputs will not be used.
        prediction_window=None, # If future inputs are provided, this will be set to the number of time steps in the future inputs.
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.num_static_inputs = num_static_inputs
        self.num_historic_inputs = num_historic_inputs
        self.num_future_inputs = num_future_inputs
        self.prediction_window = prediction_window
        self.output_size = output_size
        self.dropout_rate = dropout_rate


        if num_future_inputs is None and self.prediction_window is None:
            raise ValueError('"num_future_inputs" or "prediction_window" is required.')

        # Static
        if num_static_inputs is None:
            self.static_encoder = None
            self.static_vsn = None
        else:
            # Static covariate encoder
            self.static_encoder = StaticCovariateEncoder(
                hidden_dim=hidden_units,
                dropout_rate=dropout_rate
            )

            # Variable selection network
            self.static_vsn = VariableSelectionNetwork(
                hidden_units=hidden_units,
                num_inputs=num_static_inputs,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
                time_distributed=False
            )

        # Historic variable selection network
        self.historic_vsn = VariableSelectionNetwork(
            hidden_units=hidden_units,
            num_inputs=num_historic_inputs,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            time_distributed=True
        )

        # Future variable selection network
        if num_future_inputs is None:
            self.future_vsn = None
        else:
            self.future_vsn = VariableSelectionNetwork(
                hidden_units=hidden_units,
                num_inputs=num_future_inputs,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
                time_distributed=True
            )

        # LSTM history and future
        self.encoder_lstm = layers.LSTM(
            hidden_units,
            return_sequences=True,
            return_state=True,
            dropout=dropout_rate
        )

        self.decoder_lstm = layers.LSTM(
            hidden_units,
            return_sequences=True,
            return_state=False,
            dropout=dropout_rate
        )

        self.glu_lstm = GLU(
            hidden_units,
            dropout_rate=dropout_rate,
            time_distributed=True,
            activation=None
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
            time_distributed=True,
            activation=None
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
            time_distributed=True,
            activation=None
        )

        self.layer_norm_output = layers.LayerNormalization()

        self.output_layer = layers.TimeDistributed(layers.Dense(output_size))

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        inputs: dict with keys:
            - 'static': (batch, num_static_vars, emb_dim)
            - 'historic': (batch, time_steps, num_historic_vars, emb_dim)
            - 'future': (batch, time_steps, num_future_vars, emb_dim)
        """
        static_emb = inputs.get('static', None)
        historic_emb = inputs.get('historic', None)
        future_emb = inputs.get('future', None)

        if historic_emb is None:
            raise ValueError('Input "historic" is required.')
        
        if self.num_static_inputs is not None and static_emb is None:
            raise ValueError('Input "static" is required when "num_static_inputs" is not None.')
        
        if self.num_future_inputs is not None and future_emb is None:
            raise ValueError('Input "future" is required when "num_future_inputs" is not None.')
        

        # 1. Static covariate encoding
        if self.num_static_inputs is not None and static_emb is not None:
            static_selected = self.static_vsn(static_emb)
            static_context = self.static_encoder(static_selected)

            context_var_sel = static_context["context_variable_selection"]
            context_enrichment = static_context["context_enrichment"]
            context_state_h, context_state_c = static_context["context_state"]
        else:
            # If there are no static inputs, use zeros for context
            context_var_sel = None
            context_enrichment = context_state_h = context_state_c = ops.zeros((ops.shape(historic_emb)[0], self.hidden_units))


        # 2. Variable selection
        historic_features = self.historic_vsn(historic_emb, context=context_var_sel)

        if self.num_future_inputs is not None and future_emb is not None:
            self.prediction_window = ops.shape(future_emb)[1]
            future_features = self.future_vsn(future_emb, context=context_var_sel)
        else:
            future_features = ops.zeros((ops.shape(historic_emb)[0], self.prediction_window, self.hidden_units))


        # 3. LSTM encoder/decoder
        encoder_output, state_h, state_c = self.encoder_lstm(
            historic_features,
            initial_state=[context_state_h, context_state_c]
        )

        # Use the last state of the encoder as the initial state for the encoder
        decoder_output = self.decoder_lstm(
            future_features,
            initial_state=[state_h, state_c]
        )

        lstm_layer = ops.concatenate([encoder_output, decoder_output], axis=1)

        # Apply gated skip connection
        input_embeddings = ops.concatenate([historic_features, future_features], axis=1)

        lstm_layer, _ = self.glu_lstm(lstm_layer)
        temporal_feature_layer = ops.add(lstm_layer, input_embeddings)
        temporal_feature_layer = self.layer_norm_lstm(temporal_feature_layer)

        # 4. Static enrichment layers
        expanded_context_enrichment = ops.expand_dims(context_enrichment, axis=1)
        enriched, _ = self.grn_enrichment(
            temporal_feature_layer,
            context=expanded_context_enrichment
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