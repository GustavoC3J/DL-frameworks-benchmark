import math

import keras
from keras import initializers, layers, ops

from utils.positional_encoding import sinusoidal_encoding

# Same epsilon in the three frameworks: Keras defaults to 1e-3 and Flax to 1e-6
LN_EPSILON = 1e-5


def attention_initializer(d_model):
    """Glorot over each (d_model, d_model) projection. Keras' default would compute the fans of the
    3D kernel (d_model, heads, head_dim) with its receptive field, which Flax and torch do not."""
    limit = math.sqrt(3 / d_model)
    return initializers.RandomUniform(minval=-limit, maxval=limit)


def multi_head_attention(d_model, num_heads):
    return layers.MultiHeadAttention(
        num_heads,
        key_dim=d_model // num_heads,
        dropout=0.0, # The paper only drops the output of each sublayer
        kernel_initializer=attention_initializer(d_model),
        bias_initializer="zeros"
    )


@keras.saving.register_keras_serializable()
class PositionalEncoding(keras.Layer):

    def __init__(self, length, d_model, **kwargs):
        super().__init__(**kwargs)

        self.length = length
        self.d_model = d_model
        self.encoding = sinusoidal_encoding(length, d_model)

    def call(self, x):
        return x + ops.cast(self.encoding, self.compute_dtype)

    def get_config(self):
        return {**super().get_config(), "length": self.length, "d_model": self.d_model}


@keras.saving.register_keras_serializable()
class EncoderLayer(keras.Layer):

    def __init__(self, d_model, num_heads, d_ff, dropout, **kwargs):
        super().__init__(**kwargs)

        self.attention = multi_head_attention(d_model, num_heads)
        self.attention_dropout = layers.Dropout(dropout)
        self.attention_norm = layers.LayerNormalization(epsilon=LN_EPSILON)

        self.ffn_in = layers.Dense(d_ff, activation="relu", kernel_initializer="glorot_uniform")
        self.ffn_out = layers.Dense(d_model, kernel_initializer="glorot_uniform")
        self.ffn_dropout = layers.Dropout(dropout)
        self.ffn_norm = layers.LayerNormalization(epsilon=LN_EPSILON)

        # Config for model saving
        self.config = {"d_model": d_model, "num_heads": num_heads, "d_ff": d_ff, "dropout": dropout}

    def call(self, x, training=False):
        y = self.attention(x, x, training=training)
        x = self.attention_norm(x + self.attention_dropout(y, training=training))

        y = self.ffn_out(self.ffn_in(x))
        return self.ffn_norm(x + self.ffn_dropout(y, training=training))

    def get_config(self):
        return {**super().get_config(), **self.config}


@keras.saving.register_keras_serializable()
class DecoderLayer(keras.Layer):

    def __init__(self, d_model, num_heads, d_ff, dropout, **kwargs):
        super().__init__(**kwargs)

        self.self_attention = multi_head_attention(d_model, num_heads)
        self.self_dropout = layers.Dropout(dropout)
        self.self_norm = layers.LayerNormalization(epsilon=LN_EPSILON)

        self.cross_attention = multi_head_attention(d_model, num_heads)
        self.cross_dropout = layers.Dropout(dropout)
        self.cross_norm = layers.LayerNormalization(epsilon=LN_EPSILON)

        self.ffn_in = layers.Dense(d_ff, activation="relu", kernel_initializer="glorot_uniform")
        self.ffn_out = layers.Dense(d_model, kernel_initializer="glorot_uniform")
        self.ffn_dropout = layers.Dropout(dropout)
        self.ffn_norm = layers.LayerNormalization(epsilon=LN_EPSILON)

        self.config = {"d_model": d_model, "num_heads": num_heads, "d_ff": d_ff, "dropout": dropout}

    def call(self, x, memory, training=False):
        # Each position only attends to the previous ones
        y = self.self_attention(x, x, use_causal_mask=True, training=training)
        x = self.self_norm(x + self.self_dropout(y, training=training))

        y = self.cross_attention(x, memory, training=training)
        x = self.cross_norm(x + self.cross_dropout(y, training=training))

        y = self.ffn_out(self.ffn_in(x))
        return self.ffn_norm(x + self.ffn_dropout(y, training=training))

    def get_config(self):
        return {**super().get_config(), **self.config}


@keras.saving.register_keras_serializable()
class Transformer(keras.Model):
    """The encoder-decoder of "Attention Is All You Need", with a regression head.

    The input packs the source window and the decoder input in one array, so the runners keep a
    single input: the first `source_length` steps feed the encoder and the decoder gets the
    steps from the last source one on (the observations shifted one step, teacher forcing).
    """

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        encoder_layers,
        decoder_layers,
        dropout,
        source_length,
        horizon,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.source_length = source_length
        self.scale = math.sqrt(d_model)

        self.encoder_embedding = layers.Dense(d_model, kernel_initializer="glorot_uniform")
        self.encoder_positions = PositionalEncoding(source_length, d_model)
        self.encoder_dropout = layers.Dropout(dropout)
        self.encoder_blocks = [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(encoder_layers)]

        self.decoder_embedding = layers.Dense(d_model, kernel_initializer="glorot_uniform")
        self.decoder_positions = PositionalEncoding(horizon, d_model)
        self.decoder_dropout = layers.Dropout(dropout)
        self.decoder_blocks = [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(decoder_layers)]

        # One trip count per decoder step
        self.output_layer = layers.Dense(1, kernel_initializer="glorot_uniform")

        # Config for model saving
        self.config = {
            "d_model": d_model,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "encoder_layers": encoder_layers,
            "decoder_layers": decoder_layers,
            "dropout": dropout,
            "source_length": source_length,
            "horizon": horizon
        }

    def call(self, inputs, training=False):
        source = inputs[:, :self.source_length]
        target = inputs[:, self.source_length - 1:]

        # The embedding is scaled by sqrt(d_model), as in the paper
        x = self.encoder_positions(self.encoder_embedding(source) * self.scale)
        memory = self.encoder_dropout(x, training=training)
        for block in self.encoder_blocks:
            memory = block(memory, training=training)

        x = self.decoder_positions(self.decoder_embedding(target) * self.scale)
        x = self.decoder_dropout(x, training=training)
        for block in self.decoder_blocks:
            x = block(x, memory, training=training)

        return ops.squeeze(self.output_layer(x), axis=-1)

    def get_config(self):
        return {**super().get_config(), **self.config}
