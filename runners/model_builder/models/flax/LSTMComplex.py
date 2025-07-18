
from dataclasses import field
from typing import List, Optional

import flax.linen as nn
from flax.typing import Dtype
import jax.numpy as jnp
from jax import Array

from runners.model_builder.models.flax.tft.tft import TFT


class LSTMComplex(nn.Module):
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
    def __call__(self, inputs: Array, training=False):
        tft = TFT(
            hidden_units=self.hidden_units,
            output_size=self.output_size,
            num_attention_heads=self.num_attention_heads,
            historical_window=self.historical_window,
            prediction_window=self.prediction_window,
            observed_idx=self.observed_idx,
            unknown_idx=self.unknown_idx,
            dropout_rate=self.dropout_rate,
        )
        x = tft(inputs, training=training)   # (batch, 1, 1)
        x = x.reshape((x.shape[0], -1))  # (batch, 1)
        
        return x
