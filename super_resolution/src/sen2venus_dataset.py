"""
Data loader and utils for the SEN2VENµS dataset

Inspired by the below implementation
REF: https://github.com/piclem/sen2venus-pytorch-dataset/blob/main/sen2venus/dataset/sen2venus.py
"""

from __future__ import annotations

import warnings
import itertools
import os
import sys
import pathlib
import random
import pickle
from importlib import resources
from typing import Optional, Union, Callable

import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

try:
    import py7zr
except ImportError:
    warnings.warn(
        "Unable to import py7zr, data download functionality disabled", ImportWarning
    )

RED_INDEX = 2
GREEN_INDEX = 1
BLUE_INDEX = 0
RGB_DIMS = 3
TRAIN_PROPORTION = 0.7
VAL_PROPORTION = 0.5
CANONICAL_ORDER_RESOURCE = resources.files("super_resolution.resources").joinpath(
    "canonical_order.pkl"
)

Sample = tuple[list[str], list[str], int]
Transform = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class S2VSites:
    """Dataclass for the SEN2VENµS sites"""

    URL = "https://zenodo.org/records/6514159/files/{}.7z?download=1"
    SITES = [
        # (site name, md5sum)
        ("ALSACE", "ecbf57fc83a8c8ca47ab421642bbef57"),
        ("ANJI", "2b6521e2fd43fc220557d1a171f94c06"),
        ("ARM", "9c264cd01640707f483f78a88c1a40c8"),
        ("ATTO", "c6d7905816f8c807e5a87f4a2d09a4ae"),
        ("BAMBENW2", "f804161f30c295dab1172e904ecb38be"),
        ("BENGA", "a3bdc8fd5ac049b2d07b308fc1f0706a"),
        ("ES-IC3XG", "e7a19cd51f048a006688f6b2ea795d55"),
        ("ES-LTERA", "226cd7c10689f9aad92c760d9c1899fe"),
        ("ESGISB-1", "ab1c0e9a70c566d6fe8b94ba421a15d6"),
        ("ESGISB-2", "20196e6e963170e641fc805330077434"),
        ("ESGISB-3", "ac42ab2ddb89975b55395ace90ecc0a6"),
        ("ESTUAMAR", "2b540369499c7b9882f7e195699e9438"),
        ("FGMANAUS", "06d422d9f4ba0c2ed1087c2a7f0339c5"),
        ("FR-BIL", "c4305e091b61de5583842f71b4122ed3"),
        ("FR-LAM", "1bceb23259d7f101ee0e1df141b5e550"),
        ("FR-LQ1", "535489d0d3bc23e8e7646a20b99575e6"),
        ("JAM2018", "2e2a6de2b5842ce86d074ebd8c68354b"),
        ("K34-AMAZ", "7abf9ef3f89bd30b905c0029169b88d1"),
        ("KUDALIAR", "1427c8a4bc1e238c5c63e434fd6d31c6"),
        ("LERIDA-1", "d507dcbc1b92676410df9e4f650ea23b"),
        ("MAD-AMBO", "49e43cd47ecdc5360c83e448eaf73fbb"),
        ("NARYN", "56474220d0014e53aa0c96ea93c03bc9"),
        ("SO1", "62b5ce44dc641639079c15227cdbd794"),
        ("SO2", "59afd969b950f90df0f8ce8b1dbccd62"),
        ("SUDOUE-2", "5aed36a3d5e9746e5f5c438d10fae413"),
        ("SUDOUE-3", "0eeb556caaae171b8fbd0696f4757308"),
        ("SUDOUE-4", "aac762b62ac240720d34d5bb3fc4a906"),
        ("SUDOUE-5", "69042546af7bd25a0398b04c2ce60057"),
        ("SUDOUE-6", "ca143d2a2a56db30ab82c33420433e01"),
    ]

    @staticmethod
    def get_url(site_name: str) -> tuple[str, str]:
        for site in S2VSites.SITES:
            if site_name == site[0]:
                return (S2VSites.URL.format(site_name), site[1])

        raise ValueError(f"site {site_name} not found")

    @staticmethod
    def get_sites() -> list[str]:
        return [s[0] for s in S2VSites.SITES]


class S2VSite(Dataset):
    """SEN2VENµS dataset for a single site."""

    # blue 5m, green 5m, red 5m, NIR 5m
    VENUS_RGBNIR = "_05m_b2b3b4b8.pt"
    # red edge 5m, red edge 5m, red edge 5m, NIR 5m
    VENUS_EDGENIR = "_05m_b4b5b6b8a.pt"
    # blue 10m, green 10m, red 10m, wide NIR 10m
    SENT2_RGBNIR = "_10m_b2b3b4b8.pt"
    # 3 red edge 20m, red edge 20m, red edge 20m, narrow NIR 20m
    SENT2_EDGENIR = "_20m_b4b5b6b8a.pt"
    SCALE = 10_000

    def __init__(
        self,
        site_name: str,
        bands: str = "all",
        download_dir: str = "data/",
        device: str = "cpu",
    ):
        """
        SEN2VENµS dataset for a single site.

        Parameters:
            site_name: one of the site names in `Sen2VenusSites.get_sites()`.
            bands: one of ("all", "rgbnir", "edgenir"), defaults to "all".
            download_dir: directory to download dataset to, defaults to "data/".
            device: device to load tensors onto, defaults to "cpu".
        """
        if site_name not in S2VSites.get_sites():
            raise ValueError(f"site {site_name} not found")
        if bands not in ("all", "rgbnir", "edgenir"):
            raise ValueError(f"band group {bands} not found")

        self.site_name = site_name
        self.bands = bands

        self.url = S2VSites.get_url(site_name)
        self.download_dir = download_dir
        self.dataset_path = os.path.join(download_dir, site_name)
        self.device = device

        # Download and extract the dataset (if it hasn't already been downloaded)
        self.download_and_extract()

        # Parse samples from the extracted dataset
        self.parse_samples()

    def parse_samples(self):
        """Parse samples from extracted dataset"""
        # Each sample corresponds to a patch in the original dataset, and consists of
        # two input tensors (one for each group of bands) from Sentinel-2 and two
        # output tensors (again one for each group of bands) from VENµS.
        self.total_samples = 0
        self.samples = []

        # Extract all the pairs for this site. Each pair is uniquely identified by
        # the first three fields in each filename (site name, mgrs tile, acquisition
        # date).
        pt_files = [f for f in os.listdir(self.dataset_path) if f.endswith(".pt")]
        pair_ids = set("_".join(f.split("_")[:-2]) for f in pt_files)

        for id in pair_ids:
            # Reconstruct the input and output file names from the pair ids
            input_files = [
                os.path.join(self.dataset_path, id + self.SENT2_RGBNIR),
                os.path.join(self.dataset_path, id + self.SENT2_EDGENIR),
            ]
            target_files = [
                os.path.join(self.dataset_path, id + self.VENUS_RGBNIR),
                os.path.join(self.dataset_path, id + self.VENUS_RGBNIR),
            ]

            # Work out how many patches are in this pair; use this to keep track of the
            # total number of patches.
            num_patches = torch.load(input_files[0]).size(0)
            for batch_pos in range(num_patches):
                self.samples.append((input_files, target_files, batch_pos))
            self.total_samples += num_patches

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        input_files, target_files, pos = self.samples[index]

        if self.bands == "rgbnir":
            input_tensor = _load_sen2venus_tensor(input_files[0], pos, self.device)
            target_tensor = _load_sen2venus_tensor(target_files[0], pos, self.device)
        elif self.bands == "edgenir":
            input_tensor = _load_sen2venus_tensor(input_files[1], pos, self.device)
            target_tensor = _load_sen2venus_tensor(target_files[1], pos, self.device)
        else:
            # The Sentinel-2 EDGENIR bands are 20m in resolution, so we need to upscale
            # them to the 10m RGBNIR bands. We do this with bicubic interpolation
            input_tensor = torch.concat(
                (
                    _load_sen2venus_tensor(input_files[0], pos, self.device),
                    torch.nn.functional.interpolate(
                        _load_sen2venus_tensor(
                            input_files[1], pos, self.device
                        ).unsqueeze(0),
                        scale_factor=(2, 2),
                        mode="bicubic",
                    ).squeeze(0),
                ),
                dim=0,
            )
            target_tensor = torch.concat(
                (
                    _load_sen2venus_tensor(target_files[0], pos, self.device),
                    _load_sen2venus_tensor(target_files[1], pos, self.device),
                ),
                dim=0,
            )
        return input_tensor, target_tensor

    def is_downloaded(self) -> bool:
        """Checks if the dataset zip has already been downloaded."""
        return self.site_name + ".7z" in os.listdir(self.download_dir)

    def is_extracted(self) -> bool:
        """Checks if the dataset zip has already been extracted."""
        if not os.path.exists(self.dataset_path):
            return False
        return os.listdir(self.dataset_path) != []

    def download_and_extract(self) -> None:
        """
        Attempts to download and extract the dataset. Does not re-download nor
        re-extract if the zip and files (respectively) already exist.
        """
        if not self.is_extracted() and "py7zr" not in sys.modules:
            raise ImportError(
                "py7zr was not imported please ensure library is installed."
            )

        zip_name = self.site_name + ".7z"

        if not self.is_downloaded() and not self.is_extracted():
            download_url(
                url=self.url[0],
                root=self.download_dir,
                md5=self.url[1],
                filename=zip_name,
            )
        if not self.is_extracted():
            with py7zr.SevenZipFile(self.download_dir + zip_name, mode="r") as zip:
                zip.extractall(self.download_dir)


def default_patch_transform(
    low_res_patch: torch.Tensor, high_res_patch: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply default transform to patches. Remove 4th channel and scale RGB surface reflectant

    Parameters:
        low_res_patch (torch.Tensor): Low resolution patch
        high_res_patch (torch.Tensor): High resolution patch

    Returns
        (tuple[torch.Tensor, torch.Tensor]): The respective transformed low res patch and high res
            patch.
    """
    return (
        _min_max_scale_patch(low_res_patch[:RGB_DIMS, :, :]),
        _min_max_scale_patch(high_res_patch[:RGB_DIMS, :, :]),
    )


class PatchData(Dataset):
    """Dataset for storing patch file data."""

    def __init__(
        self,
        samples: list[Sample],
        device: Union[torch.device, str] = "cpu",
        transform: Transform = default_patch_transform,
    ):
        """
        Parameters:
            samples (list[Sample]): Patch samples.
            device (torch.device | str): Device to load tensors to. Default is cpu.
            transform (callable): Transform to apply to each patch before getting.
        """
        self.samples = samples
        self.device = device
        self._transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_files, target_files, pos = self.samples[index]

        input_tensor, target_tensor = self._transform(
            _load_sen2venus_tensor(input_files[0], pos, self.device),
            _load_sen2venus_tensor(target_files[0], pos, self.device),
        )

        return input_tensor, target_tensor

    def set_transform(self, transform: Transform) -> None:
        """Set the patch transform."""
        self._transform = transform


def download_all_site_data(download_dir: str) -> None:
    """Download and extracts all site data into the given download directory."""
    for site_name, _ in S2VSites.SITES:
        print(f"Downloading site {site_name}")
        S2VSite(
            site_name=site_name,
            bands="rgbnir",
            download_dir=download_dir,
            device="cpu",
        )


def create_train_test_split(
    data_dir: str,
    seed: int = -1,
    sites: Optional[set[str]] = None,
    device: Union[torch.device, str] = "cpu",
) -> tuple[PatchData, PatchData]:
    """Create train-test split using satellite data.

    Parameters:
        data_dir (str): Directory where all site data is downloaded.
        seed (int): Seed to randomly shuffle data. Default is -1 which uses canonical ordering.
        sites (Optional[set[str]]): Set of sites to create split from. Default is None which
            creates split from all sites.
        device: (torch.device | str): Device to load tensors to. Default is cpu.

    Returns:
        (Optional[tuple[PatchData, PatchData]]): train dataset, test dataset tuple.
    """
    data_dir_path = pathlib.Path(data_dir)
    downloaded_sites = _get_downloaded_sites(data_dir_path)
    all_sites = sites if sites else set(site_name for site_name, _ in S2VSites.SITES)
    missing = all_sites - downloaded_sites

    # Download if required
    if len(missing) != 0 and _check_to_download(len(all_sites), len(missing)):
        download_all_site_data(data_dir)
    sites = _get_downloaded_sites(data_dir_path) & all_sites

    # Gather all samples
    site_samples = [[([""], [""], -1)] for _ in sites]
    for i, site_name in enumerate(sites):
        site = S2VSite(
            site_name=site_name,
            bands="rgbnir",
            download_dir=data_dir,
            device="cpu",
        )
        site_samples[i] = site.samples
    all_samples = sorted(itertools.chain.from_iterable(site_samples))

    # Reorder all samples
    if seed == -1:
        canonical_order = _load_canonical_order()
        all_samples = [all_samples[i] for i in canonical_order if i < len(all_samples)]
    else:
        random.seed(seed)
        random.shuffle(all_samples)

    cut_off = int(TRAIN_PROPORTION * len(all_samples))
    train_patches = PatchData(all_samples[:cut_off], device=device)
    test_patches = PatchData(all_samples[cut_off:], device=device)
    return train_patches, test_patches


def create_train_validation_test_split(
    data_dir: str,
    seed: int = -1,
    sites: Optional[set[str]] = None,
    device: Union[torch.device, str] = "cpu",
) -> tuple[PatchData, PatchData, PatchData]:
    """Create train-validation-test split using satellite data.

    Parameters:
        data_dir (str): Directory where all site data is downloaded.
        seed (int): Seed to randomly shuffle data. Default is -1 which uses canonical ordering.
        sites (Optional[set[str]]): Set of sites to create split from. Default is None which
            creates split from all sites.
        device: (torch.device | str): Device to load tensors to. Default is cpu.

    Returns:
        (Optional[tuple[PatchData, PatchData, PatchData]]): train dataset, validation dataset,
        test dataset tuple.
    """
    train, test = create_train_test_split(
        data_dir, seed=seed, sites=sites, device=device
    )
    cut_off = int(VAL_PROPORTION * len(test.samples))
    return (
        train,
        PatchData(test.samples[:cut_off], device=device),
        PatchData(test.samples[cut_off:], device=device),
    )


def _check_to_download(total: int, num_missing: int) -> bool:
    response = ""
    while len(response) != 1 or response not in "YyNn":
        response = input(
            f"Missing {num_missing}/{total} sites data."
            " Would you like to download now? Yes (Y/y) or no (N/n): "
        )
    return response in "yY"


def _get_downloaded_sites(data_dir: pathlib.Path) -> set[str]:
    available_sites = set(site_name for site_name, _ in S2VSites.SITES)
    downloaded_sites = set(path.stem for path in data_dir.iterdir() if path.is_dir())
    return available_sites & downloaded_sites


def _load_sen2venus_tensor(
    file: str, pos: int, device: Union[torch.device, str]
) -> torch.Tensor:
    patch = torch.load(file, map_location=device)[pos] / S2VSite.SCALE
    return patch[[RED_INDEX, GREEN_INDEX, BLUE_INDEX]]


def _load_canonical_order() -> list[int]:
    buffer = CANONICAL_ORDER_RESOURCE.read_bytes()
    return pickle.loads(buffer)


def _min_max_scale_patch(patch: torch.Tensor) -> torch.Tensor:
    min_val = patch.min()
    max_val = patch.max()
    return (patch - min_val) / (max_val - min_val)
