from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from natsort import natsorted
from rich.console import Console
from rich.progress import Progress

from src.models.components.clip_models import build_clip_model
from src.utils.io import pack_list_of_np_arrays

CONSOLE = Console()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_caption_model(clip_cfg_path: str):
    with open(clip_cfg_path) as f:
        clip_cfg = yaml.safe_load(f)
    clip_cfg = edict(clip_cfg["clip_encoder"])
    clip_model = build_clip_model(clip_cfg, device=device)
    clip_model.eval()
    return clip_model


@torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
@torch.inference_mode()
def extract_text_embeddings(
    caption_dir: str, clip_cfg_path: str, caption_file: str = "captions.npz"
):
    CONSOLE.print(
        f"Extracting text embeddings from {caption_dir}, "
        f"clip_cfg_path: {clip_cfg_path}, "
        f"caption_file: {caption_file}"
    )

    clip_model = load_caption_model(clip_cfg_path)
    clip_model_name = Path(clip_cfg_path).stem

    caption_dir = Path(caption_dir)
    scene_dirs = natsorted(list(caption_dir.iterdir()))
    error_files = []
    with Progress(console=CONSOLE) as progress:
        task = progress.add_task("Extracting text embeddings", total=len(scene_dirs))
        for scene_dir in scene_dirs:
            scene_name = scene_dir.stem
            caption_file_path = scene_dir / caption_file
            if not caption_file_path.exists():
                CONSOLE.print(f"Caption file not found for {scene_name}")
                error_files.append(scene_name)
                progress.advance(task)
                continue

            CONSOLE.print(f"Processing {scene_name}")
            data = np.load(caption_file_path)
            captions = data["packed"]
            num_captions = data["lengths"]

            CONSOLE.print(f"Extracting embeddings for {len(captions)} captions")
            tokens = clip_model.text_tokenizer(captions).to(device)
            embeddings = clip_model.encode_text(tokens)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = embeddings.half().cpu().numpy()

            output_file = scene_dir / caption_file.replace("captions", "embeddings").replace(
                ".npz", f".{clip_model_name}.npz"
            )
            chunked_embeddings = np.split(embeddings, np.cumsum(num_captions)[:-1])
            packed = pack_list_of_np_arrays(chunked_embeddings)
            CONSOLE.print(f"Saving embeddings to {output_file}")
            np.savez_compressed(output_file, **packed)
            progress.advance(task)


if __name__ == "__main__":
    import fire

    fire.Fire(extract_text_embeddings)
