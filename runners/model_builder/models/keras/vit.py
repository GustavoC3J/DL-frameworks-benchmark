
import keras
from keras import layers, ops


class Patches(layers.Layer):

    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        # Same result as keras.ops.image.extract_patches, which is a convolution: reshapes are what torch and Flax use
        _, height, width, channels = images.shape
        rows, cols = height // self.patch_size, width // self.patch_size

        patches = ops.reshape(images, (-1, rows, self.patch_size, cols, self.patch_size, channels))
        patches = ops.transpose(patches, (0, 1, 3, 2, 4, 5))

        # (batch, patches, pixels of the patch in row, column and channel order)
        return ops.reshape(patches, (-1, rows * cols, self.patch_size * self.patch_size * channels))

    def get_config(self):
        super_config = super().get_config()
        super_config.update({"patch_size": self.patch_size})
        return super_config



class ClassToken(layers.Layer):

    def __init__(self, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.token = self.add_weight(
            name="class_token",
            shape=(1, 1, self.projection_dim),
            initializer="zeros"
        )

    def call(self, patches):
        # Broadcast over the batch without reading its size, which graph mode leaves undefined
        token = ops.zeros_like(patches[:, :1, :]) + self.token

        return ops.concatenate([token, patches], axis=1)

    def get_config(self):
        super_config = super().get_config()
        super_config.update({"projection_dim": self.projection_dim})
        return super_config



class PatchEncoder(layers.Layer):

    def __init__(self, num_patches, projection_dim, dropout = 0.1, **kwargs):
        super().__init__(**kwargs)

        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.dropout_rate = dropout

        self.projection = layers.Dense(projection_dim)
        self.class_token = ClassToken(projection_dim)

        # One position more than patches: the class token gets its own
        self.position_embedding = layers.Embedding(
            input_dim=num_patches + 1,
            output_dim=projection_dim,
            embeddings_initializer=keras.initializers.RandomNormal(stddev=0.02)
        )
        self.dropout = layers.Dropout(dropout)

    def call(self, patches, training = False):
        x = self.class_token(self.projection(patches))
        positions = ops.expand_dims(ops.arange(self.num_patches + 1), axis=0)

        return self.dropout(x + self.position_embedding(positions), training=training)

    def get_config(self):
        super_config = super().get_config()
        super_config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
            "dropout": self.dropout_rate
        })
        return super_config



class TransformerBlock(layers.Layer):

    def __init__(
        self,
        projection_dim,
        num_heads,
        mlp_dim,
        dropout = 0.1,
        attention_dropout = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim // num_heads, dropout=attention_dropout)
        self.attention_dropout = layers.Dropout(dropout)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = layers.Dense(mlp_dim, activation="gelu")
        self.dropout1 = layers.Dropout(dropout)
        self.dense2 = layers.Dense(projection_dim)
        self.dropout2 = layers.Dropout(dropout)

        # Config for model saving
        self.config = {
            "projection_dim": projection_dim,
            "num_heads": num_heads,
            "mlp_dim": mlp_dim,
            "dropout": dropout,
            "attention_dropout": attention_dropout
        }


    def call(self, x, training = False):
        # Pre-norm: each sublayer sees the normalized input and adds its output to the residual
        y = self.norm1(x)
        y = self.attention(y, y, training=training)
        x = x + self.attention_dropout(y, training=training)

        y = self.norm2(x)
        y = self.dropout1(self.dense1(y), training=training)
        y = self.dropout2(self.dense2(y), training=training)

        return x + y

    def get_config(self):
        super_config = super().get_config()
        super_config.update(self.config)
        return super_config





# Since Keras 3.9, load_model only imports keras modules, so custom models must be registered
@keras.saving.register_keras_serializable()
class ViT(keras.Model):

    def __init__(
        self,
        image_size,
        patch_size,
        projection_dim,
        num_heads,
        transformer_layers,
        mlp_dim,
        num_classes,
        dropout = 0.1,
        attention_dropout = 0.0,
        **kwargs
    ):
        """
        image_size: Height and width of the square input images
        patch_size: Height and width of each patch
        projection_dim: Size of the patch representations
        num_heads: Attention heads, each of size projection_dim // num_heads
        transformer_layers: Number of transformer blocks
        mlp_dim: Hidden units of the MLP of each block
        num_classes: Number of classes
        dropout: Dropout rate after the position embeddings and every dense layer
        attention_dropout: Dropout rate of the attention weights
        """
        super().__init__(**kwargs)

        num_patches = (image_size // patch_size) ** 2

        self.patches = Patches(patch_size)
        self.patch_encoder = PatchEncoder(num_patches, projection_dim, dropout)

        self.transformer_blocks = [
            TransformerBlock(projection_dim, num_heads, mlp_dim, dropout, attention_dropout)
            for _ in range(transformer_layers)
        ]

        self.norm = layers.LayerNormalization(epsilon=1e-6)

        # Output layer
        self.classifier = layers.Dense(num_classes, activation="softmax")


        # Config for model saving
        self.config = {
            "image_size": image_size,
            "patch_size": patch_size,
            "projection_dim": projection_dim,
            "num_heads": num_heads,
            "transformer_layers": transformer_layers,
            "mlp_dim": mlp_dim,
            "num_classes": num_classes,
            "dropout": dropout,
            "attention_dropout": attention_dropout
        }


    def call(self, inputs, training = False):
        x = self.patch_encoder(self.patches(inputs), training=training)

        for block in self.transformer_blocks:
            x = block(x, training=training)

        # The state of the class token is the representation of the image
        return self.classifier(self.norm(x)[:, 0])


    def get_config(self):
        super_config = super().get_config()
        super_config.update(self.config)
        return super_config
