
from flax import linen as nn
import jax.numpy as jnp
from typing import Sequence, Optional


class Block(nn.Module):
    in_channels: int
    out_channels: int
    stride: int
    
    dtype: any
    param_dtype: any

    @nn.compact
    def __call__(self, x, training):
        residual = x

        x = nn.Conv(
            self.out_channels,
            kernel_size=3,
            strides=self.stride,
            use_bias=False, # Not needed before batchnorm
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)

        x = nn.Conv(
            self.out_channels,
            kernel_size=3,
            strides=1,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)

        # Downsample si hace falta
        if self.stride > 1 or self.in_channels != self.out_channels:
            residual = nn.Conv(
                self.out_channels,
                kernel_size=1,
                strides=self.stride,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype
            )(residual)

        x = x + residual
        x = nn.relu(x)

        return x
    

class CNNComplex(nn.Module):
    n_blocks: int
    
    dtype: any
    param_dtype: any

    @nn.compact
    def __call__(self, x, training):
        in_channels = 16

        x = nn.Conv(
            in_channels,
            kernel_size=3,
            use_bias=False, # Not needed before batchnorm
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)

        # Each stage is composed of n blocks whose convolutions use the corresponding filters
        filters = [16, 32, 64]
        for stage, out_channels in enumerate(filters):
            for i in range(self.n_blocks):
                # If it is the first block of the stage, a subsampling is made
                stride = 2 if stage > 0 and i == 0 else 1
                x = Block(in_channels, out_channels, stride, self.dtype, self.param_dtype)(x, training)
                in_channels = out_channels

        # Flatten and perform final prediction
        x = jnp.mean(x, axis=(2, 3))  # Global average pooling
        x = nn.Dense(10, dtype=self.dtype, param_dtype=self.param_dtype)(x) # softmax is applied in loss function

        return x