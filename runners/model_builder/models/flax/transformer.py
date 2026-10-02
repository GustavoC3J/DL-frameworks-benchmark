import math

import flax.linen as nn
import jax.numpy as jnp

from utils.positional_encoding import sinusoidal_encoding

# Same epsilon in the three frameworks: Keras defaults to 1e-3 and Flax to 1e-6
LN_EPSILON = 1e-5


def multi_head_attention(d_model, num_heads, dtype, param_dtype, training):
    # Flax flattens the kernels to (d_model, d_model) before initializing them
    return nn.MultiHeadDotProductAttention(
        num_heads=num_heads,
        qkv_features=d_model,
        dtype=dtype,
        param_dtype=param_dtype,
        kernel_init=nn.initializers.glorot_uniform(),
        bias_init=nn.initializers.zeros,
        dropout_rate=0.0, # The paper only drops the output of each sublayer
        broadcast_dropout=False, # Each attention weight on its own, as in Keras and torch
        deterministic=not training
    )


def dense(features, dtype, param_dtype):
    return nn.Dense(features, kernel_init=nn.initializers.glorot_uniform(), dtype=dtype, param_dtype=param_dtype)


def layer_norm(dtype, param_dtype):
    return nn.LayerNorm(epsilon=LN_EPSILON, dtype=dtype, param_dtype=param_dtype)


class EncoderLayer(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    dropout: float
    dtype: any
    param_dtype: any

    @nn.compact
    def __call__(self, x, training):
        y = multi_head_attention(self.d_model, self.num_heads, self.dtype, self.param_dtype, training)(x, x)
        x = layer_norm(self.dtype, self.param_dtype)(x + nn.Dropout(self.dropout)(y, deterministic=not training))

        y = dense(self.d_ff, self.dtype, self.param_dtype)(x)
        y = dense(self.d_model, self.dtype, self.param_dtype)(nn.relu(y))
        return layer_norm(self.dtype, self.param_dtype)(x + nn.Dropout(self.dropout)(y, deterministic=not training))


class DecoderLayer(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    dropout: float
    dtype: any
    param_dtype: any

    @nn.compact
    def __call__(self, x, memory, causal_mask, training):
        # Each position only attends to the previous ones. In a Flax mask, True means "attend"
        y = multi_head_attention(self.d_model, self.num_heads, self.dtype, self.param_dtype, training)(x, x, mask=causal_mask)
        x = layer_norm(self.dtype, self.param_dtype)(x + nn.Dropout(self.dropout)(y, deterministic=not training))

        y = multi_head_attention(self.d_model, self.num_heads, self.dtype, self.param_dtype, training)(x, memory)
        x = layer_norm(self.dtype, self.param_dtype)(x + nn.Dropout(self.dropout)(y, deterministic=not training))

        y = dense(self.d_ff, self.dtype, self.param_dtype)(x)
        y = dense(self.d_model, self.dtype, self.param_dtype)(nn.relu(y))
        return layer_norm(self.dtype, self.param_dtype)(x + nn.Dropout(self.dropout)(y, deterministic=not training))


class Transformer(nn.Module):
    """The encoder-decoder of "Attention Is All You Need", with a regression head.

    The input packs the source window and the decoder input in one array, so the runners keep a
    single input: the first `source_length` steps feed the encoder and the decoder gets the
    steps from the last source one on (the observations shifted one step, teacher forcing).
    """
    d_model: int
    num_heads: int
    d_ff: int
    encoder_layers: int
    decoder_layers: int
    dropout: float
    source_length: int
    dtype: any
    param_dtype: any

    @nn.compact
    def __call__(self, x, training):
        source = x[:, :self.source_length]
        target = x[:, self.source_length - 1:]
        scale = math.sqrt(self.d_model)

        positions = lambda length: jnp.asarray(sinusoidal_encoding(length, self.d_model), self.dtype)

        # The embedding is scaled by sqrt(d_model), as in the paper
        memory = dense(self.d_model, self.dtype, self.param_dtype)(source) * scale + positions(source.shape[1])
        memory = nn.Dropout(self.dropout)(memory, deterministic=not training)
        for _ in range(self.encoder_layers):
            memory = EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout, self.dtype, self.param_dtype)(memory, training)

        horizon = target.shape[1]
        causal_mask = jnp.tril(jnp.ones((1, 1, horizon, horizon), dtype=bool))

        x = dense(self.d_model, self.dtype, self.param_dtype)(target) * scale + positions(horizon)
        x = nn.Dropout(self.dropout)(x, deterministic=not training)
        for _ in range(self.decoder_layers):
            x = DecoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout, self.dtype, self.param_dtype)(x, memory, causal_mask, training)

        # One trip count per decoder step. Linear: a ReLU here kills the network at the start of training
        return dense(1, self.dtype, self.param_dtype)(x)[..., 0]
