
import torch
import torch.nn as nn

from utils.torch_utils import init_layer_weights


class Patches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, images):
        # The images stay NHWC, so each patch is flattened in row, column and channel order, as in Keras and Flax
        batch, height, width, channels = images.shape
        rows, cols = height // self.patch_size, width // self.patch_size

        patches = images.reshape(batch, rows, self.patch_size, cols, self.patch_size, channels)
        patches = patches.permute(0, 1, 3, 2, 4, 5)

        return patches.reshape(batch, rows * cols, self.patch_size * self.patch_size * channels)



class ClassToken(nn.Module):
    def __init__(self, projection_dim):
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, 1, projection_dim))

    def forward(self, patches):
        return torch.cat([self.token.expand(patches.shape[0], -1, -1), patches], dim=1)



class PatchEncoder(nn.Module):
    def __init__(self, num_patches, patch_dim, projection_dim, dropout = 0.1):
        super().__init__()

        self.projection = nn.Linear(patch_dim, projection_dim)
        init_layer_weights(self.projection, "glorot_uniform")

        self.class_token = ClassToken(projection_dim)

        # One position more than patches: the class token gets its own
        self.position_embedding = nn.Embedding(num_patches + 1, projection_dim)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

        self.dropout = nn.Dropout(dropout)

    def forward(self, patches):
        x = self.class_token(self.projection(patches))
        positions = torch.arange(self.position_embedding.num_embeddings, device=patches.device)

        return self.dropout(x + self.position_embedding(positions))



class TransformerBlock(nn.Module):
    def __init__(self, projection_dim, num_heads, mlp_dim, dropout = 0.1, attention_dropout = 0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(projection_dim, eps=1e-6)

        self.attention = nn.MultiheadAttention(projection_dim, num_heads, dropout=attention_dropout, batch_first=True)
        # Keras' glorot_uniform for each projection: torch draws Q, K and V as one matrix and the output as a Linear
        for weight in self.attention.in_proj_weight.chunk(3):
            nn.init.xavier_uniform_(weight)
        init_layer_weights(self.attention.out_proj, "glorot_uniform")

        self.attention_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(projection_dim, eps=1e-6)

        dense1 = nn.Linear(projection_dim, mlp_dim)
        dense2 = nn.Linear(mlp_dim, projection_dim)
        init_layer_weights(dense1, "glorot_uniform")
        init_layer_weights(dense2, "glorot_uniform")

        self.mlp = nn.Sequential(dense1, nn.GELU(), nn.Dropout(dropout), dense2, nn.Dropout(dropout))

    def forward(self, x):
        y = self.norm1(x)
        # need_weights=False lets torch use the fused scaled_dot_product_attention
        y, _ = self.attention(y, y, y, need_weights=False)
        x = x + self.attention_dropout(y)

        return x + self.mlp(self.norm2(x))



class ViT(nn.Module):
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
        channels = 3
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
        channels: Channels of the input images
        """
        super().__init__()

        num_patches = (image_size // patch_size) ** 2

        self.patches = Patches(patch_size)
        self.patch_encoder = PatchEncoder(num_patches, patch_size * patch_size * channels, projection_dim, dropout)

        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(projection_dim, num_heads, mlp_dim, dropout, attention_dropout)
            for _ in range(transformer_layers)
        ])

        self.norm = nn.LayerNorm(projection_dim, eps=1e-6)

        # Output layer
        self.classifier = nn.Linear(projection_dim, num_classes)
        init_layer_weights(self.classifier, "glorot_uniform")


    def forward(self, x):
        x = self.patch_encoder(self.patches(x))
        x = self.transformer_blocks(x)

        # The state of the class token is the representation of the image
        return self.classifier(self.norm(x)[:, 0])
