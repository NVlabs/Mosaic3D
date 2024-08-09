import fire

from src.data.download_utils import download_hf


def download(download_dir: str):
    download_hf(
        repo_id="Pointcept/scannet-compressed",
        file="scannet.tar.gz",
        download_dir=download_dir,
    )


if __name__ == "__main__":
    fire.Fire(download)
