"""
Data loader and utils for the SEN2VENµS dataset
"""

import os
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url


class Sen2VenusSites:
    """Dataclass for the Sen2Venus sites"""

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
        for site in Sen2VenusSites.SITES:
            if site_name == site[0]:
                return (Sen2VenusSites.URL.format(site_name), site[1])

        raise ValueError(f"site {site_name} not found")

    @staticmethod
    def get_sites() -> list[str]:
        return [s[0] for s in Sen2VenusSites.SITES]


class Sen2VenusDataset(Dataset):
    """SEN2VENµS dataset."""

    BAND_SUFFIXES = [
        "_05m_b2b3b4b8.pt",  # venus blue 5m, green 5m, red 5m, NIR 5m
        "_05m_b4b5b6b8a.pt",  # venus red edge 5m, red edge 5m, red edge 5m, NIR 5m
        "_10m_b2b3b4b8.pt",  # sentinel-2 blue 10m, green 10m, red 10m, wide NIR 10m
        "_20m_b4b5b6b8a.pt",  # sentinel-2 red edge 20m, red edge 20m, red edge 20m, narrow NIR 20m
    ]

    def __init__(self, site_name: str, bands: str = "all", download_dir: str = "data/"):
        """
        Parameters:
            site_name: one of the site names in `Sen2VenusSites.get_sites()`.
            bands: one of ("all", "rgbnir", "edgenir"), defaults to "all".
            download_dir: directory to download dataset to, defaults to "data/".
        """
        if site_name not in Sen2VenusSites.get_sites():
            raise ValueError(f"site {site_name} not found")
        if bands not in ("all", "rgbnir", "edgenir"):
            raise ValueError(f"band group {bands} not found")

        self.site_name = site_name
        self.download_dir = download_dir
        self.dataset_path = os.path.join(download_dir, site_name)
        self.url = Sen2VenusSites.get_url(site_name)

        # Download and extract the dataset (if it hasn't already been downloaded)
        self.download_and_extract()

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
        import py7zr

        zip_name = self.site_name + ".7z"

        if not self.is_downloaded():
            download_url(
                url=self.url[0],
                root=self.download_dir,
                md5=self.url[1],
                filename=zip_name,
            )
        if not self.is_extracted():
            with py7zr.SevenZipFile(self.download_dir + zip_name, mode="r") as zip:
                zip.extractall(self.download_dir)
