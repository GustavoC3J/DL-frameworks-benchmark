
import flax.linen as nn
import jax.numpy as jnp


class Patches(nn.Module):
    patch_size: int

    def __call__(self, images):
        # Each patch is flattened in row, column and channel order, as in Keras and torch
        batch, height, width, channels = images.shape
        rows, cols = height // self.patch_size, width // self.patch_size

        patches = images.reshape(batch, rows, self.patch_size, cols, self.patch_size, channels)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)

        return patches.reshape(batch, rows * cols, self.patch_size * self.patch_size * channels)



class ClassToken(nn.Module):
    projection_dim: int

    dtype: any
    param_dtype: any

    @nn.compact
    def __call__(self, patches):
        token = self.param("token", nn.initializers.zeros, (1, 1, self.projection_dim), self.param_dtype)
        token = jnp.broadcast_to(token.astype(self.dtype), (patches.shape[0], 1, self.projection_dim))

        return jnp.concatenate([token, patches], axis=1)



class PatchEncoder(nn.Module):
    num_patches: int
    projection_dim: int

    dtype: any
    param_dtype: any

    dropout: float = 0.1

    @nn.compact
    def __call__(self, patches, training):
        x = nn.Dense(
            self.projection_dim,
            kernel_init=nn.initializers.glorot_uniform(),
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )(patches)

        x = ClassToken(
            projection_dim=self.projection_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )(x)

        # One position more than patches: the class token gets its own
        position_embedding = nn.Embed(
            self.num_patches + 1,
            self.projection_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        x = x + position_embedding(jnp.arange(self.num_patches + 1))

        return nn.Dropout(self.dropout)(x, deterministic=not training)



class TransformerBlock(nn.Module):
    projection_dim: int
    num_heads: int
    mlp_dim: int

    dtype: any
    param_dtype: any

    dropout: float = 0.1
    attention_dropout: float = 0.0

    @nn.compact
    def __call__(self, x, training):
        y = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, param_dtype=self.param_dtype)(x)

        # By default Flax shares the attention dropout mask across the batch; Keras and torch draw one per element
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.attention_dropout,
            broadcast_dropout=False,
            kernel_init=nn.initializers.glorot_uniform(),
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )(y, deterministic=not training)
        y = nn.Dropout(self.dropout)(y, deterministic=not training)
        x = x + y

        y = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        y = nn.Dense(self.mlp_dim, kernel_init=nn.initializers.glorot_uniform(), dtype=self.dtype, param_dtype=self.param_dtype)(y)
        # Exact GELU, as in Keras and torch: Flax defaults to the tanh approximation
        y = nn.gelu(y, approximate=False)
        y = nn.Dropout(self.dropout)(y, deterministic=not training)
        y = nn.Dense(self.projection_dim, kernel_init=nn.initializers.glorot_uniform(), dtype=self.dtype, param_dtype=self.param_dtype)(y)
        y = nn.Dropout(self.dropout)(y, deterministic=not training)

        return x + y



class ViT(nn.Module):
    image_size: int
    patch_size: int
    projection_dim: int
    num_heads: int
    transformer_layers: int
    mlp_dim: int
    num_classes: int

    dtype: any
    param_dtype: any

    dropout: float = 0.1
    attention_dropout: float = 0.0

    @nn.compact
    def __call__(self, x, training):
        num_patches = (self.image_size // self.patch_size) ** 2

        x = Patches(self.patch_size)(x)
        x = PatchEncoder(
            num_patches=num_patches,
            projection_dim=self.projection_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            dropout=self.dropout
        )(x, training)

        for _ in range(self.transformer_layers):
            x = TransformerBlock(
                projection_dim=self.projection_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                dropout=self.dropout,
                attention_dropout=self.attention_dropout
            )(x, training)

        x = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, param_dtype=self.param_dtype)(x)

        # The state of the class token is the representation of the image
        x = x[:, 0]

        return nn.Dense(
            self.num_classes,
            kernel_init=nn.initializers.glorot_uniform(),
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )(x) # softmax is applied in loss function
