import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any


class ConvBlock(nn.Module):
    """A simple 2D convolutional block with optional batch normalization and ReLU activation."""
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        return x


class DownBlock(nn.Module):
    """A downsampling block with max pooling followed by a ConvBlock."""
    features: int

    @nn.compact
    def __call__(self, x):
        skip = ConvBlock(self.features)(x)
        x = nn.max_pool(skip, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        return x, skip


class UpBlock(nn.Module):
    """An upsampling block with transposed convolution followed by a ConvBlock."""
    features: int

    @nn.compact
    def __call__(self, x, skip):
        x = nn.ConvTranspose(self.features, kernel_size=(2, 2), strides=(2, 2), padding='SAME')(x)
        x = jnp.concatenate([x, skip], axis=-1)
        x = ConvBlock(self.features)(x)
        return x


class UNet(nn.Module):
    """The U-Net model."""
    features: int = 64

    @nn.compact
    def __call__(self, x):
        # Encoder
        x, skip1 = DownBlock(self.features)(x)
        x, skip2 = DownBlock(self.features * 2)(x)
        x, skip3 = DownBlock(self.features * 4)(x)
        x, skip4 = DownBlock(self.features * 8)(x)

        # Bottleneck
        x = ConvBlock(self.features * 16)(x)

        # Decoder
        x = UpBlock(self.features * 8)(x, skip4)
        x = UpBlock(self.features * 4)(x, skip3)
        x = UpBlock(self.features * 2)(x, skip2)
        x = UpBlock(self.features)(x, skip1)

        # Output layer
        #x = nn.Conv(self.num_classes, kernel_size=(1, 1), padding='SAME')(x)
        x = x.reshape((x.shape[0], x.shape[1]*x.shape[2], x.shape[3]))
        x = jnp.mean(x, axis = 1)
        return x


if(__name__ == "__main__"):
    from jax import random

    # Create an instance of the UNet model
    model = UNet()

    # Initialize the model with random parameters
    rng = random.PRNGKey(0)
    input_shape = (1, 128, 128, 3)  # Example input shape (batch_size, height, width, channels)
    variables = model.init(rng, jnp.ones(input_shape))

    # Run a forward pass
    output = model.apply(variables, jnp.ones(input_shape))
    print(output.shape)  # Expected output shape: (1, 128, 128, num_classes)
