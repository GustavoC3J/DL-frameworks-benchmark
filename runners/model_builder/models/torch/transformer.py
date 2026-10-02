import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.positional_encoding import sinusoidal_encoding
from utils.torch_utils import init_attention_weights, init_layer_weights

# Same epsilon in the three frameworks: Keras defaults to 1e-3 and Flax to 1e-6
LN_EPSILON = 1e-5


def multi_head_attention(d_model, num_heads):
    attention = nn.MultiheadAttention(
        d_model,
        num_heads,
        dropout=0.0, # The paper only drops the output of each sublayer
        batch_first=True
    )
    init_attention_weights(attention)

    return attention


def linear(in_features, out_features):
    layer = nn.Linear(in_features, out_features)
    init_layer_weights(layer, "glorot_uniform")

    return layer


class PositionalEncoding(nn.Module):

    def __init__(self, length, d_model):
        super().__init__()

        # A constant, not a parameter: kept out of the state_dict
        self.register_buffer("encoding", torch.from_numpy(sinusoidal_encoding(length, d_model)), persistent=False)

    def forward(self, x):
        # Same dtype as the input, or autocast would promote the whole stream to float32
        return x + self.encoding.to(x.dtype)


class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        # Sublayers in the order of the data flow: the equivalence tests pair layers by position
        self.attention = multi_head_attention(d_model, num_heads)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(d_model, eps=LN_EPSILON)

        self.ffn_in = linear(d_model, d_ff)
        self.ffn_out = linear(d_ff, d_model)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model, eps=LN_EPSILON)

    def forward(self, x):
        # need_weights=False takes the fused scaled_dot_product_attention path
        y, _ = self.attention(x, x, x, need_weights=False)
        x = self.attention_norm(x + self.attention_dropout(y))

        y = self.ffn_out(F.relu(self.ffn_in(x)))
        return self.ffn_norm(x + self.ffn_dropout(y))


class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.self_attention = multi_head_attention(d_model, num_heads)
        self.self_dropout = nn.Dropout(dropout)
        self.self_norm = nn.LayerNorm(d_model, eps=LN_EPSILON)

        self.cross_attention = multi_head_attention(d_model, num_heads)
        self.cross_dropout = nn.Dropout(dropout)
        self.cross_norm = nn.LayerNorm(d_model, eps=LN_EPSILON)

        self.ffn_in = linear(d_model, d_ff)
        self.ffn_out = linear(d_ff, d_model)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model, eps=LN_EPSILON)

    def forward(self, x, memory, causal_mask):
        # Each position only attends to the previous ones. In a boolean mask, True means "do not attend"
        y, _ = self.self_attention(x, x, x, attn_mask=causal_mask, is_causal=True, need_weights=False)
        x = self.self_norm(x + self.self_dropout(y))

        y, _ = self.cross_attention(x, memory, memory, need_weights=False)
        x = self.cross_norm(x + self.cross_dropout(y))

        y = self.ffn_out(F.relu(self.ffn_in(x)))
        return self.ffn_norm(x + self.ffn_dropout(y))


class Transformer(nn.Module):
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
        features=11
    ):
        super().__init__()

        self.source_length = source_length
        self.scale = math.sqrt(d_model)

        # Layers in the order of the data flow: the equivalence tests pair layers by position
        self.encoder_embedding = linear(features, d_model)
        self.encoder_positions = PositionalEncoding(source_length, d_model)
        self.encoder_dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(encoder_layers)])

        self.decoder_embedding = linear(features, d_model)
        self.decoder_positions = PositionalEncoding(horizon, d_model)
        self.decoder_dropout = nn.Dropout(dropout)
        self.decoder_blocks = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(decoder_layers)])

        # One trip count per decoder step. Linear: a ReLU here kills the network at the start of training
        self.output_layer = linear(d_model, 1)

        causal_mask = torch.triu(torch.ones(horizon, horizon, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, inputs):
        source = inputs[:, :self.source_length]
        target = inputs[:, self.source_length - 1:]

        # The embedding is scaled by sqrt(d_model), as in the paper
        x = self.encoder_positions(self.encoder_embedding(source) * self.scale)
        memory = self.encoder_dropout(x)
        for block in self.encoder_blocks:
            memory = block(memory)

        x = self.decoder_positions(self.decoder_embedding(target) * self.scale)
        x = self.decoder_dropout(x)
        for block in self.decoder_blocks:
            x = block(x, memory, self.causal_mask)

        return self.output_layer(x).squeeze(-1)
