"""Module for testing."""

from typing import Callable
from dataclasses import dataclass

import torch
from torcheval.metrics.functional import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader


@dataclass
class Metr"""Module for testing."""

from typing import Callable
from dataclasses import dataclass

import torch
from torcheval.metrics.functional import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader


@dataclass
class Metrics:
    """Super-resolution metrics"""

    mse: float
    psnr: float
    ssim: float


def compute_metrics(
    super_resolver, x,y
):
    """Compute super resolution metrics using the given super-resolution model.

    Parameters:
        super_resolver (Callable[[torch.Tensor], torch.Tensor]): Super-resolution model.
            Must be callable and accept a samples x channels x width x height tensor of low
                resolution images and return high resolution images.
        loader (DataLoader): Image loader, should iterate over low resolution, high_resolution
            batches.
    """
    mean_psnr = 0.0
    mean_ssim = 0.0
    mean_mse = 0.0
    
    for low_res, high_res in loader:
        with torch.no_grad():
            super_resolved = super_resolver(low_res)
            mean_mse += mse_loss(super_resolved, high_res).item()
            mean_psnr += peak_signal_noise_ratio(super_resolved, high_res, 1.0).item()
            ssim = structural_similarity_index_measure(
                super_resolved, high_res, data_range=(0.0, 1.0)
            )
            assert isinstance(ssim, torch.Tensor)
            mean_ssim += ssim.item()
    mean_mse /= len(loader)
    mean_ssim /= len(loader)
    mean_psnr /= len(loader)

    return Metrics(mean_mse, mean_psnr, mean_ssim)
    """Super-resolution metrics"""
