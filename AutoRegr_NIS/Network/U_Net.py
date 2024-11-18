import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any
from .MLPs import ReluMLP

class ConvBlock(nn.Module):
    """A simple 2D convolutional block with optional batch normalization and ReLU activation."""
    features: int
    padding: str = "CIRCULAR"
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features, kernel_size=(3, 3), padding=self.padding)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, kernel_size=(3, 3), padding=self.padding)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        return x


class DownBlock(nn.Module):
    """A downsampling block with max pooling followed by a ConvBlock."""
    features: int
    padding: str = "VALID"
    @nn.compact
    def __call__(self, x):
        skip = ConvBlock(self.features)(x)
        x = nn.max_pool(skip, window_shape=(2, 2), strides=(2, 2), padding=self.padding)
        return x, skip


class UpBlock(nn.Module):
    """An upsampling block with transposed convolution followed by a ConvBlock."""
    features: int
    padding: str = "CIRCULAR"
    def crop_and_concat(self, upsampled, skip):
        """Crop the skip connection to match the size of the upsampled feature map."""
        # Calculate the amount of cropping needed
        crop_h = (skip.shape[1] - upsampled.shape[1]) // 2
        crop_w = (skip.shape[2] - upsampled.shape[2]) // 2

        skip_cropped = skip[:, crop_h:crop_h + upsampled.shape[1], crop_w:crop_w + upsampled.shape[2], :]

        return jnp.concatenate([upsampled, skip_cropped], axis=-1)

    @nn.compact
    def __call__(self, x, skip):
        #print("conv_transpose", x.shape)
        x = nn.ConvTranspose(self.features, kernel_size=(2, 2), strides=(2, 2), padding=self.padding)(x)
        #print("conv_transpose after", x.shape)
        x = self.crop_and_concat(x, skip)
        x = ConvBlock(self.features)(x)
        return x


class UNet(nn.Module):
    """The U-Net model."""
    n_layers: int = 1
    features: int = 64

    @nn.compact
    def __call__(self, x):
        # Encoder
        x = ReluMLP(n_features_list=[self.features], dtype = jnp.float32)(x)
        #print("RELU",x.shape)
        pow = 1
        skip_features = []
        for n_layer in range(self.n_layers):
            x, skip1 = DownBlock(self.features*pow)(x)
            pow *= 2
            skip_features.append(skip1)
            #print("RELU down",x.shape)

        # Bottleneck
        x = ConvBlock(self.features * pow)(x)

        # Decoder
        # x = UpBlock(self.features * 8)(x, skip4)
        # x = UpBlock(self.features * 4)(x, skip3)
        # x = UpBlock(self.features * 2)(x, skip2)
        # x = UpBlock(self.features)(x, skip1)

        for idx  in range(self.n_layers):
            pow = int(pow/2)
            #print("RELU up",x.shape, skip_features[-1-idx].shape)
            x = UpBlock(self.features * pow)(x, skip_features[-1-idx])
            #print("RELU after",x.shape)

        # Output layer
        #x = nn.Conv(self.num_classes, kernel_size=(1, 1), padding='SAME')(x)
        #print("RELU2",x.shape)
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
