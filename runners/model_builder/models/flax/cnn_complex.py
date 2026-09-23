
from flax import linen as nn
import jax.numpy as jnp
from typing import Callable, Sequence, Optional

# Same values in the three frameworks: Keras defaults to an epsilon of 1e-3
BN_MOMENTUM = 0.99
BN_EPSILON = 1e-5


class Block(nn.Module):
    in_channels: int
    out_channels: int
    stride: int
    kernel_initializer: Callable
    
    dtype: any
    param_dtype: any


    @nn.compact
    def __call__(self, x, training):
        residual = x

        x = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            use_bias=False, # Not needed before batchnorm
            kernel_init=self.kernel_initializer,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )(x)
        x = nn.BatchNorm(
            momentum=BN_MOMENTUM,
            epsilon=BN_EPSILON,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            # Keras and torch keep the statistics in the policy's dtype; Flax forces float32 by default
            force_float32_reductions=False
        )(x, use_running_average=not training)
        x = nn.relu(x)

        x = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            use_bias=False,
            kernel_init=self.kernel_initializer,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )(x)
        x = nn.BatchNorm(
            momentum=BN_MOMENTUM,
            epsilon=BN_EPSILON,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            force_float32_reductions=False
        )(x, use_running_average=not training)

        # Downsample if needed
        if self.stride > 1 or self.in_channels != self.out_channels:
            residual = nn.Conv(
                self.out_channels,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                use_bias=False,
                kernel_init=self.kernel_initializer,
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
    
    starting_channels: int = 16
    kernel_initializer: Callable = nn.initializers.he_uniform

    @nn.compact
    def __call__(self, x, training):

        initializer = self.kernel_initializer(dtype=self.dtype)

        x = nn.Conv(
            self.starting_channels,
            kernel_size=(3, 3),
            use_bias=False, # Not needed before batchnorm
            kernel_init=initializer,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )(x)
        x = nn.BatchNorm(
            momentum=BN_MOMENTUM,
            epsilon=BN_EPSILON,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            force_float32_reductions=False
        )(x, use_running_average=not training)
        x = nn.relu(x)

        # Each stage is composed of n blocks whose convolutions use the corresponding filters
        in_channels = self.starting_channels
        filters = [16, 32, 64]

        for stage, out_channels in enumerate(filters):
            for i in range(self.n_blocks):
                # If it is the first block of the stage, a subsampling is made
                stride = 2 if stage > 0 and i == 0 else 1
                x = Block(in_channels, out_channels, stride, initializer, self.dtype, self.param_dtype)(x, training)
                in_channels = out_channels

        # Flatten and perform final prediction
        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        x = nn.Dense(100, kernel_init=nn.initializers.glorot_uniform(), dtype=self.dtype, param_dtype=self.param_dtype)(x) # softmax is applied in loss function

        return x