"""Adapted from 
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution/blob/master/models.py
"""

import pathlib
import math

import torch
import torchvision
from torch import nn


class ConvolutionalBlock(nn.Module):
    """A convolutional block, comprising convolutional, BN, activation layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride=1,
        batch_norm=False,
        activation=None,
    ):
        """Initialize a Convolutional Block.


        Parameters:
            in_channels: number of input channels
            out_channels: number of output channe;s
            kernel_size: kernel size
            stride: stride
            batch_norm: include a BN layer?
            activation: Type of activation; None if none
        """
        super().__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {"prelu", "leakyrelu", "tanh"}

        layers: list[nn.Module] = []

        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
        )

        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        if activation == "prelu":
            layers.append(nn.PReLU())
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU(0.2))
        elif activation == "tanh":
            layers.append(nn.Tanh())

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Forward propagation.

        Parameters:
            input: input images, a tensor of size (N, in_channels, w, h)

        Returns:
            (torch.Tensor): output images, a tensor of size (N, out_channels, w, h)
        """
        output = self.conv_block(x)

        return output


class SubPixelConvolutionalBlock(nn.Module):
    """A subpixel convolutional block, comprising convolutional, pixel-shuffle, and PReLU activation
    layers.
    """

    def __init__(
        self, kernel_size: int = 3, n_channels: int = 64, scaling_factor: int = 2
    ):
        """
        Parameters:
            kernel_size: kernel size of the convolution
            n_channels: number of input and output channels
            scaling_factor: factor to scale input images by (along both dimensions)
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels * (scaling_factor**2),
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor):
        """
        Parameters:
            input: input images, a tensor of size (N, n_channels, w, h)

        Returns:
            scaled output images, a tensor of size
                (N, n_channels, w * scaling factor, h * scaling factor)
        """
        # (N, n_channels * scaling factor^2, w, h)
        output = self.conv(x)

        # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.pixel_shuffle(output)

        # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)

        return output


class ResidualBlock(nn.Module):
    """A residual block, comprising two convolutional blocks with a residual connection across
    them.
    """

    def __init__(self, kernel_size: int = 3, n_channels: int = 64):
        """
        Parameters:
            kernel_size: kernel size
            n_channels: number of input and output channels
                (same because the input must be added to the output)
        """
        super().__init__()

        # The first convolutional block
        self.conv_block1 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            batch_norm=False,
            activation="PReLu",
        )

        # The second convolutional block
        self.conv_block2 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            batch_norm=False,
            activation=None,
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters:
            input: input images, a tensor of size (N, n_channels, w, h)
        Returns:
            (torch.Tensor): output images, a tensor of size (N, n_channels, w, h)
        """
        # (N, n_channels, w, h)
        residual = x

        # (N, n_channels, w, h)
        output = self.conv_block1(x)

        # (N, n_channels, w, h)
        output = self.conv_block2(output)

        # (N, n_channels, w, h)
        output = output + residual

        return output


class SRResNet(nn.Module):
    """The SRResNet, as defined in the paper."""

    def __init__(
        self,
        large_kernel_size: int = 9,
        small_kernel_size: int = 3,
        n_channels: int = 64,
        n_blocks: int = 16,
        scaling_factor: int = 2,
    ):
        """
        Parameters:
            large_kernel_size: kernel size of the first and last convolutions which transform the
                inputs and outputs
            small_kernel_size: kernel size of all convolutions in-between, i.e. those in the
                residual and subpixel convolutional blocks
            n_channels: number of channels in-between, i.e. the input and output channels for the
                residual and subpixel convolutional blocks
            n_blocks: number of residual blocks
            scaling_factor: factor to scale input images by (along both dimensions) in the subpixel
                convolutional block
        """
        super().__init__()

        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8}, "The scaling factor must be 2, 4, or 8!"

        self.conv_block1 = ConvolutionalBlock(
            in_channels=3,
            out_channels=n_channels,
            kernel_size=large_kernel_size,
            batch_norm=False,
            activation="PReLu",
        )

        # Each contains a skip-connection across the block
        blocks = [
            ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels)
            for _ in range(n_blocks)
        ]
        self.residual_blocks = nn.Sequential(*blocks)

        self.conv_block2 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=small_kernel_size,
            batch_norm=False,
            activation=None,
        )

        # Upscaling
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        upscale_blocks = [
            SubPixelConvolutionalBlock(
                kernel_size=small_kernel_size,
                n_channels=n_channels,
                scaling_factor=2,
            )
            for _ in range(n_subpixel_convolution_blocks)
        ]
        self.subpixel_convolutional_blocks = nn.Sequential(*upscale_blocks)

        self.conv_block3 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=3,
            kernel_size=large_kernel_size,
            batch_norm=False,
            activation="Tanh",
        )

    def forward(self, lr_imgs: torch.Tensor):
        """
        Parameters:
            lr_imgs: low-resolution input images, a tensor of size (N, 3, w, h)

        Returns:
            (torch.Tensor): super-resolution output images, a tensor of size
            (N, 3, w * scaling factor, h * scaling factor)
        """
        # (N, 3, w, h)
        output = self.conv_block1(lr_imgs)

        # (N, n_channels, w, h)
        residual = output

        # (N, n_channels, w, h)
        output = self.residual_blocks(output)

        # (N, n_channels, w, h)
        output = self.conv_block2(output)

        # (N, n_channels, w, h)
        output = output + residual

        # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.subpixel_convolutional_blocks(output)

        # (N, 3, w * scaling factor, h * scaling factor)
        sr_imgs = self.conv_block3(output)

        return sr_imgs


class Generator(nn.Module):
    """The generator in the SRGAN, as defined in the paper. Architecture identical to the
    SRResNet.
    """

    def __init__(
        self,
        large_kernel_size: int = 9,
        small_kernel_size: int = 3,
        n_channels: int = 64,
        n_blocks: int = 16,
        scaling_factor: int = 2,
    ):
        """
        Parameters:
            large_kernel_size: kernel size of the first and last convolutions which transform the
                inputs and outputs
            small_kernel_size: kernel size of all convolutions in-between, i.e. those in the
                residual and subpixel convolutional blocks
            n_channels: number of channels in-between, i.e. the input and output channels for the
                residual and subpixel convolutional blocks
            n_blocks: number of residual blocks
            scaling_factor: factor to scale input images by (along both dimensions) in the subpixel
                convolutional block
        """
        super().__init__()

        self.net = SRResNet(
            large_kernel_size=large_kernel_size,
            small_kernel_size=small_kernel_size,
            n_channels=n_channels,
            n_blocks=n_blocks,
            scaling_factor=scaling_factor,
        )

    def initialize_with_srresnet(self, srresnet_checkpoint: pathlib.Path):
        """
        Initialize with weights from a trained SRResNet.

        Parameters:
            srresnet_checkpoint: checkpoint filepath
        """
        srresnet = torch.load(srresnet_checkpoint)["model"]
        self.net.load_state_dict(srresnet.state_dict())

        print("\nLoaded weights from pre-trained SRResNet.\n")

    def forward(self, lr_imgs: torch.Tensor):
        """
        Parameters:
            lr_imgs: low-resolution input images, a tensor of size (N, 3, w, h)

        Returns:
            super-resolution output images, a tensor of size
            (N, 3, w * scaling factor, h * scaling factor)
        """
        # (N, n_channels, w * scaling factor, h * scaling factor)
        sr_imgs = self.net(lr_imgs)

        return sr_imgs


class Discriminator(nn.Module):
    """
    The discriminator in the SRGAN, as defined in the paper.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        n_channels: int = 64,
        n_blocks: int = 8,
        fc_size: int = 1024,
    ):
        """
        Parameters:
            kernel_size: kernel size in all convolutional blocks
            n_channels: number of output channels in the first convolutional block, after which
                it is doubled in every 2nd block thereafter
            n_blocks: number of convolutional blocks
            fc_size: size of the first fully connected layer
        """
        super().__init__()

        in_channels = 3

        # A series of convolutional blocks
        # The first, third, fifth (and so on) convolutional blocks increase the number of channels
        # but retain image size
        # The second, fourth, sixth (and so on) convolutional blocks retain the same number of
        # channels but halve image size
        # The first convolutional block is unique because it does not employ batch normalization
        conv_blocks = []
        for i in range(n_blocks):
            out_channels = (
                (n_channels if i == 0 else in_channels * 2)
                if i % 2 == 0
                else in_channels
            )
            conv_blocks.append(
                ConvolutionalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1 if i % 2 == 0 else 2,
                    batch_norm=i != 0,
                    activation="LeakyReLu",
                )
            )
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # An adaptive pool layer that resizes it to a standard size
        # For the default input size of 96 and 8 convolutional blocks, this will have no effect
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, imgs):
        """
        Forward propagation.

        :param imgs: high-resolution or super-resolution images which must be classified as such, a
            tensor of size (N, 3, w * scaling factor, h * scaling factor)
        :return: a score (logit) for whether it is a high-resolution image, a tensor of size (N)
        """
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)

        return logit


class TruncatedVGG19(nn.Module):
    """
    A truncated VGG19 network, such that its output is the
    'feature map obtained by the j-th convolution (after activation)
    before the i-th maxpooling layer within the VGG19 network', as defined in the paper.

    Used to calculate the MSE loss in this VGG feature-space, i.e. the VGG loss.
    """

    def __init__(self, i: int = 5, j: int = 4):
        """
        Parameters:
            i: the index i in the definition above
            j: the index j in the definition above
        """
        super().__init__()

        # Load the pre-trained VGG19 available in torchvision
        vgg19 = torchvision.models.vgg19(weights="DEFAULT")

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        # Iterate through the convolutional section ("features") of the VGG19
        for layer in vgg19.features.children():
            truncate_at += 1

            # Count the number of maxpool layers and the convolutional layers after each maxpool
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # Break if we reach the jth convolution after the (i - 1)th maxpool
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # Check if conditions were satisfied
        assert (
            maxpool_counter == i - 1 and conv_counter == j
        ), f"One or both of i={i} and j={j} are not valid choices for the VGG19!"

        # Truncate to the jth convolution (+ activation) before the ith maxpool layer
        self.truncated_vgg19 = nn.Sequential(
            *list(vgg19.features.children())[: truncate_at + 1]
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters:
            input: high-resolution or super-resolution images, a tensor of size
                (N, 3, w * scaling factor, h * scaling factor)

        Returns
            (torch.Tensor): the specified VGG19 feature map, a tensor of size
                (N, feature_map_channels, feature_map_w, feature_map_h)
        """
        # (N, feature_map_channels, feature_map_w, feature_map_h)
        output = self.truncated_vgg19(x)

        return output
