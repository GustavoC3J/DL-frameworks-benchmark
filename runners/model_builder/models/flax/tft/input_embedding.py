
from dataclasses import field
from typing import List, Optional

import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype
from jax import Array


class InputEmbedding(nn.Module):
    """Embedding layer for TFT model."""

    num_inputs: int
    embedding_dim: int
    historical_window: int
    prediction_window: int
    static_idx: List[int]
    observed_idx: List[int]
    known_idx: List[int]
    unknown_idx: List[int]
    categorical_idx: List[int] = field(default_factory=list)
    categorical_counts: List[int] = field(default_factory=list)
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: Array, *, training: bool = False):
        """
        input: Tensor of shape (batch_size, window, num_inputs)

        Static: Static features
        Historical: Observed, known and unknown features
        Future: Known features
        """

        input_shape = inputs.shape
        batch_size = input_shape[0]
        window_size = self.historical_window + self.prediction_window

        # If inputs' window is longer than needed, slice it
        if input_shape[1] > window_size:
            inputs = inputs[:, input_shape[1] - window_size :, :]

        # Build embedding layers for each input
        static_inputs = []
        historical_inputs = []
        future_inputs = []

        cat_map = {idx: count for idx, count in zip(self.categorical_idx, self.categorical_counts)}
        
        for i in range(self.num_inputs):
            single_input = inputs[:, :, i:i+1]  # (batch_size, window, 1)

            if i in self.categorical_idx:
                count = cat_map[i]

                if count <= 0:
                    raise ValueError(f'Invalid count {count} for categorical index {i}. Must be greater than 0.')
                
                embed_layer = nn.Embed(
                    num_embeddings=count,
                    features=self.embedding_dim,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    name=f'cat_embedding_{i}',
                )

                output = embed_layer(single_input)  # (batch_size, window, embedding_dim)
            else:
                dense_layer = nn.Dense(
                    features=self.embedding_dim,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    name=f'dense_embedding_{i}',
                )

                output = dense_layer(single_input)  # (batch_size, window, embedding_dim)

            if output.ndim == 3:
                output = jnp.expand_dims(output, axis=2)

            # For static, only keep the first timestep
            if i in self.static_idx:
                static_inputs.append(output[:, 0:1, :, :])

            elif i in self.known_idx:
                historical_inputs.append(output[:, :self.historical_window, :, :])
                future_inputs.append(output[:, self.historical_window:, :, :])

            else:
                historical_inputs.append(output[:, :self.historical_window, :])
        
        # Join each list into a tensor (batch_size, num_static, embedding_dim)
        static_inputs = jnp.concatenate(static_inputs, axis=-2) if static_inputs else None

        # (batch, window, num observed + unknown + known, embedding_dim)
        historical_inputs = jnp.concatenate(historical_inputs, axis=-2)

        # (batch, pred_window, num_known, embedding_dim)
        future_inputs = jnp.concatenate(future_inputs, axis=-2) if future_inputs else None

        return static_inputs, historical_inputs, future_inputs
