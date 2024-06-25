import os
from concurrent.futures import ThreadPoolExecutor

import requests
from huggingface_hub import hf_hub_download
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


def download_hf(repo_id: str, file: str, download_dir: str):
    hf_hub_download(
        repo_id=repo_id,
        filename=file,
        repo_type="dataset",
        local_dir=download_dir,
        local_dir_use_symlinks=False,
    )
    if file.endswith(".tar.gz"):
        os.system(f"tar -xzf {download_dir}/{file} -C {download_dir}")
        os.system(f"rm {download_dir}/{file}")
    elif file.endswith(".zip"):
        os.system(f"unzip {download_dir}/{file} -d {download_dir}")
        os.system(f"rm {download_dir}/{file}")
    else:
        print("File extension not supported.")


def download_single_file(
    url,
    headers,
    filename,
    progress,
    task_id,
    overwrite: bool = False,
):
    local_filename = filename
    if not overwrite and os.path.exists(local_filename):
        print(f"Already exists: {local_filename}")
        return

    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total_length = r.headers.get("content-length")
        if total_length is None:
            with open(local_filename, "wb") as f:
                f.write(r.content)
        else:
            total_length = int(total_length)
            progress.update(task_id, total=total_length)
            with open(local_filename, "wb") as f:
                progress.start_task(task_id)
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))
    progress.console.log(f"Downloaded {local_filename}")
    return local_filename


def download_parallel(urls, headers, filenames, max_workers=5, overwrite: bool = False):
    progress = Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
    )

    with progress:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for url, filename in zip(urls, filenames):
                task_id = progress.add_task("download", filename=filename, start=False)
                executor.submit(
                    download_single_file,
                    url,
                    headers,
                    filename,
                    progress,
                    task_id,
                    overwrite,
                )
