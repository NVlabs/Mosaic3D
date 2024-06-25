# download RegionPLC dataset
import os

import fire
import requests

from src.data.download_utils import download_hf, download_parallel


def get_and_save(url, headers, save_path):
    response = requests.get(url, headers=headers)
    with open(save_path, "wb") as f:
        f.write(response.content)


def download(download_dir: str = "/root/data/regionplc"):
    # download preprocessed scannet from Pointcept
    if not os.path.exists(os.path.join(download_dir)):
        download_hf("Pointcept/scannet-compressed", "scannet.tar.gz", download_dir=download_dir)

    # download captions & corr info
    # TODO: UPDATE THIS VALUE TO RECENT ONE ##
    cookies = {
        "FedAuth": "77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjEzLDBoLmZ8bWVtYmVyc2hpcHx1cm4lM2FzcG8lM2Fhbm9uI2MwZmEyZDAyMGZlMWI5N2YyYmIxYTJlM2QwODViYWUwZDdmZWNlZmNlMDg3NzQ0NzEwYmRlN2Q2OTJmMTk4NjYsMCMuZnxtZW1iZXJzaGlwfHVybiUzYXNwbyUzYWFub24jYzBmYTJkMDIwZmUxYjk3ZjJiYjFhMmUzZDA4NWJhZTBkN2ZlY2VmY2UwODc3NDQ3MTBiZGU3ZDY5MmYxOTg2NiwxMzM2MzczODE4NDAwMDAwMDAsMCwxMzM2MzgyNDI4NDI0ODgxOTAsMC4wLjAuMCwyNTgsZTgwZDhlNzUtNTJiOS00ODM5LWEzNTgtODdhYmI5M2IzNTY3LCwsZmVlNDM1YTEtMjA0Yi0zMDAwLTViMDktZmQ1N2M1NTUyNjQ3LGZlZTQzNWExLTIwNGItMzAwMC01YjA5LWZkNTdjNTU1MjY0NyxsMG14MUpTS3dVU2NPNElVdUc2eHFRLDAsMCwwLCwsLDI2NTA0Njc3NDM5OTk5OTk5OTksMCwsLCwsLCwwLCwxOTYwMDAsRGFEQWZqUVFtcHlPWHgyUnJLX1c1bHZvTFo0LEpLRmN3MkhuZjh4NzNJQUE0QWZOQzdiNVhjbEZrNUFLYVJNZDVKMTJRUCt0dUJoM3JaM1JMRUNwWi9UdnpvL3A5OGVwNmV2bi9RdGVBNlQzUXNtMmlraTRCVzBQSy9OQ0o4dllrKzhmRTZnK3R5MUxqdEU5OW5nRnJ0ajBOYkRhREpEMFdkNHp0Y3hVY2F4QVdnYTV6ZVBNY3k3WEFXQktZVldtYkVxVUVaeFhkOEw2am16M2VvQU40Nll2VHRzdFBUTGhHN2lTVFZ6WjR0RVF3MTZ3NGhqOHFvZXlTTDdFakQvZUE5SjZqaUZrUEx4UDJQZTNxV0lWWDNPUnFOdXc2bFVKSEtNd3QxeklzazB5U2c3WThycHRNZ2RTa0NrVlp4ZFI5b09EbXhvZ1ZmVnhxVzRvdHQ5OXhQTUxOWDVkai9TM2JXZStiQ2dkUjVGM0NPbmFRUT09PC9TUD4=",
    }

    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "en-US,en;q=0.9",
        "priority": "u=0, i",
        "referer": "https://connecthkuhk-my.sharepoint.com/personal/jhyang13_connect_hku_hk/_layouts/15/onedrive.aspx?view=0&id=%2Fpersonal%2Fjhyang13%5Fconnect%5Fhku%5Fhk%2FDocuments%2Fpretrained%5Fmodels%2Fregionplc%2Fcaption%5Ffiles%2Fscannet",
        "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "iframe",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "service-worker-navigation-preload": '{"supportsFeatures":[]}',
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "cookie": "; ".join([f"{k}={v}" for k, v in cookies.items()]),
    }

    ids_and_filenames = [
        (
            "2d092411%2D0554%2D4978%2Daa94%2D670940b633d9",
            "caption/caption_detic_and_sw_125k_iou0.3.json",
        ),
        (
            "47909841%2D4653%2D4b93%2Da33f%2D91ec825a7526",
            "caption/caption_detic-template_and_kosmos_125k_iou0.2.json",
        ),
        (
            "80553734%2D06bf%2D4b2a%2D8afa%2D292df5f17c12",
            "caption/caption_kosmos_and_detic-template_125k_iou0.2.json",
        ),
        (
            "e9174ac0%2D65dc%2D45c0%2D9e22%2D7956e34ee6e8",
            "caption/caption_kosmos_and_sw_125k_iou0.2-0.0.json",
        ),
        (
            "6ee3c0d7%2Dd3f7%2D4c77%2Dadfa%2D438d83035ece",
            "caption/caption_kosmos_and_sw_25k_iou0.1.json",
        ),
        (
            "7c9226bb%2Df876%2D473c%2D8dd3%2D6a39aece8a5b",
            "caption/caption_kosmos_and_sw_and_detic-template_125k_iou0.2_0.1.json",
        ),
        (
            "f02f127b%2De589%2D4b8a%2Da12f%2D5665e972036c",
            "image_corr/scannet_caption_idx_detic_and_sw_125k_iou0.3.pkl",
        ),
        (
            "0dc4cd55%2D164d%2D492c%2D81cd%2D44f32f7d0666",
            "image_corr/scannet_caption_idx_detic-template_and_kosmos_125k_iou0.2.pkl",
        ),
        (
            "3024f62a%2D8366%2D4b24%2D921e%2Dddb5ebe7fe3f",
            "image_corr/scannet_caption_idx_kosmos_and_detic-template_125k_iou0.2.pkl",
        ),
        (
            "886c1e28%2Df87b%2D4faa%2Dbafd%2D93391d352a84",
            "image_corr/scannet_caption_idx_kosmos_and_sw_125k_iou0.2-0.0.pkl",
        ),
        (
            "bfc350c1%2D762c%2D41f4%2D84ed%2D05362873fe32",
            "image_corr/scannet_caption_idx_kosmos_and_sw_25k_iou0.1.pkl",
        ),
        (
            "1204634c%2D48da%2D4ac7%2D825d%2Df4046fba37cf",
            "image_corr/scannet_caption_idx_kosmos_and_sw_and_detic-template_125k_iou0.2_0.1.pkl",
        ),
        ("6e0f41a2%2Dd61d%2D442c%2Da119%2Ddd5276add2a9", "text_embed/scannet_clip-ViT-B16_id.pth"),
        (
            "8437e358%2D912e%2D4474%2Da7e5%2D9105f90559d4",
            "text_embed/scannet_clip-ViT-B32_lseg.pth",
        ),
        (
            "09235430%2D0ad1%2D401c%2Db907%2Dd678abd08cdd",
            "text_embed/scannet200_clip-ViT-B16_id.pth",
        ),
    ]

    baseurl = "https://connecthkuhk-my.sharepoint.com/personal/jhyang13_connect_hku_hk/_layouts/15/download.aspx"
    download_urls, filenames = zip(
        *[
            (
                f"{baseurl}?UniqueId={uid}",
                os.path.join(download_dir, filename),
            )
            for uid, filename in ids_and_filenames
        ]
    )
    download_parallel(urls=download_urls, headers=headers, filenames=filenames, max_workers=6)


if __name__ == "__main__":
    fire.Fire(download)
