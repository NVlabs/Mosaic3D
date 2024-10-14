import os
import gdown

ID_CKPTS = {
    "segment3d.ckpt": "1Swq9d7rjV2Q1lTuXiKh1z0OZPt9V4sgO",
}
ID_SAMPLES = [
    "1Ffe5jcItYmZHpb875Z3V48wsqsAAnDp9",
    "1pKmMo9zp9PUUdz2VO3fFNHNbdtu1nDcH",
]


def download_ckpt():
    if not os.path.exists("ckpts"):
        os.makedirs("ckpts", exist_ok=True)

    for name, id_ckpt in ID_CKPTS.items():
        if os.path.exists(f"ckpts/{name}"):
            print(f"ckpt {name} already exists")
            continue
        gdown.download(
            f"https://drive.google.com/uc?id={id_ckpt}",
            f"ckpts/{name}",
            quiet=False,
        )


def download_sample():
    if not os.path.exists("samples"):
        os.makedirs("samples", exist_ok=True)

    for id_sample in ID_SAMPLES:
        if os.path.exists(f"samples/{id_sample}"):
            print(f"sample {id_sample} already exists")
            continue
        gdown.download_folder(
            f"https://drive.google.com/drive/folders/{id_sample}",
            output=f"samples/{id_sample}",
            quiet=False,
            use_cookies=False,
        )


if __name__ == "__main__":
    download_ckpt()
    download_sample()
