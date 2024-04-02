"""
Data loader and utils for the SEN2VENµS dataset
"""


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
