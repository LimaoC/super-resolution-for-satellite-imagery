"""Module for testing."""

from typing import Callable
from dataclasses import dataclass

import tqdm
import torch
from torcheval.metrics import FrechetInceptionDistance
from torcheval.metrics.functional import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

COLUMN_WIDTH = 100


@dataclass
class Metrics:
    """Super-resolution metrics"""

    mse: float
    psnr: float
    ssim: float
    fid: float


def compute_metrics(
    super_resolver: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    verbose: bool = True,
):
    """Compute super resolution metrics using the given super-resolution model.

    Parameters:
        super_resolver (Callable[[torch.Tensor], torch.Tensor]): Super-resolution model.
            Must be callable and accept a samples x channels x width x height tensor of low
                resolution images and return high resolution images.
        loader (DataLoader): Image loader, should iterate over low resolution, high_resolution
            batches.
        verbose (bool): True to show progress.
    """
    if verbose:
        loop = tqdm.tqdm(loader, ncols=COLUMN_WIDTH, total=len(loader))
    else:
        loop = loader

    mean_psnr = 0.0
    mean_ssim = 0.0
    mean_mse = 0.0
    frechet = FrechetInceptionDistance()

    for low_res, high_res in loop:
        with torch.no_grad():
            super_resolved = super_resolver(low_res)
            mean_mse += mse_loss(super_resolved, high_res).item()
            mean_psnr += peak_signal_noise_ratio(super_resolved, high_res, 1.0).item()
            ssim = structural_similarity_index_measure(
                super_resolved, high_res, data_range=(0.0, 1.0)
            )
            assert isinstance(ssim, torch.Tensor)
            mean_ssim += ssim.item()

            super_resolved = super_resolved.clamp(0, 1)
            frechet.update(super_resolved, False)
            frechet.update(high_res, True)

    mean_mse /= len(loader)
    mean_ssim /= len(loader)
    mean_psnr /= len(loader)
    fid = frechet.compute()

    return Metrics(mean_mse, mean_psnr, mean_ssim, fid.item())
