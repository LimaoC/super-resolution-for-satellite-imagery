"""Visualization utilities for super resolution."""

import math
import pathlib
from typing import Optional

import torch
import matplotlib.pyplot as plt

CMAPS = {2: "gray", 3: None}


def plot_gallery(
    images: list[torch.Tensor],
    titles: Optional[list[str]] = None,
    xscale: float = 1.5,
    yscale: float = 1.5,
    nrow: int = 1,
    output: Optional[pathlib.Path] = None,
) -> None:
    """Plot the given images in a gallery.

    Parameters:
        images (torch.Tensor): Images tensor with dimensions WxHxC.
        titles (Optional[list[str]]): Titles for each each image
        xscale (float): Width of each image plot in inches
        yscale (float): Height of image plot in inches
        nrow (int): The number of rows
        output (Optional[pathlib.Path]): Path to save output to.
    """
    ncol = math.ceil(len(images) / nrow)

    plt.figure(figsize=(xscale * ncol, yscale * nrow))

    for i in range(min(nrow * ncol, len(images))):
        image = images[i]
        cmap = CMAPS[len(image.shape)]
        plt.subplot(nrow, ncol, i + 1)
        plt.imshow(image, cmap=cmap)

        if titles is not None:
            # use size and y to adjust font size and position of title
            plt.title(titles[i], size=10, y=1)

        plt.xticks(())
        plt.yticks(())

    plt.tight_layout()

    if output is not None:
        plt.savefig(output)

    plt.show()
