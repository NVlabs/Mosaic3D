from typing import List, Tuple

import hashlib
import os
import abc
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel


# Use a deterministic hash function for strings
def string_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


class CLIPTextEncoderInterace(abc.ABC):
    model: torch.nn.Module
    CHANNEL_DIM: int

    def __post_init__(self):
        self.freeze_encoder()

    def freeze_encoder(self):
        for params in self.model.parameters():
            params.requires_grad = False

    @abc.abstractmethod
    def __call__(self, list_of_texts: List[str], normalize: bool = True) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def get_unique_text_embedding(
        self,
        list_of_texts: List[str] | List[List[str]],
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get unique embeddings for a list of texts.

        Args:
            list_of_texts: List[str] | List[List[str]]
                List of texts or list of list of texts to get unique embeddings for.
                Total number of texts is N.

        Returns:
            embeddings: torch.Tensor, shape (M, D)
                Unique embeddings for the list of texts.
            from_unique_indices: torch.Tensor, shape (N,)
                Indices of the texts in the original list.
            to_unique_indices: torch.Tensor, shape (M,)
                Indices of the unique texts in the flattened list.
        """
        # Flatten the list of texts
        if isinstance(list_of_texts, list) and isinstance(list_of_texts[0], list):
            # list of lists
            list_of_texts = [item for sublist in list_of_texts for item in sublist]

        # cchoy: Get unique texts using hash. Using string directly is not deterministic due to python string object not using the string values only for hashing.
        flat_caption_hash = [string_hash(caption) for caption in list_of_texts]
        _, to_unique_indices, from_unique_indices = np.unique(
            flat_caption_hash, return_index=True, return_inverse=True
        )

        # Get unique texts
        unique_texts = [list_of_texts[i] for i in to_unique_indices]

        # Get embeddings
        embeddings = self(unique_texts, normalize=normalize)

        # Return embeddings and indices
        return embeddings, torch.tensor(from_unique_indices), torch.tensor(to_unique_indices)


def get_text_encoder(
    model_type: str,
    device: str,
    **kwargs,
) -> CLIPTextEncoderInterace:
    if model_type == "siglip2":
        return Siglip2TextEncoder(device=device, **kwargs)
    elif model_type == "openclip":  # Recap CLIP is also openclip
        return OpenCLIPTextEncoder(device=device, **kwargs)
    else:
        raise ValueError(f"Model type {model_type} not supported")


class OpenCLIPTextEncoder(CLIPTextEncoderInterace):
    CHANNEL_DIM = None

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        # This is a not a required dependency, so we need to import it here
        try:
            from open_clip import create_model_from_pretrained, get_tokenizer
        except ImportError:
            raise ImportError(
                "open_clip is not installed. Please install it with `pip install open-clip`"
            )
        self.prepare_data(model_id)

        self.tokenizer = get_tokenizer(model_id)
        precision = {torch.float16: "fp16", torch.bfloat16: "bf16"}[torch_dtype]
        self.model, _ = create_model_from_pretrained(
            model_id,
            device=device,
            precision=precision,
        )
        self.device = device

    def prepare_data(self, model_id: str):
        from open_clip.factory import download_pretrained_from_hf

        # Remove hf-hub: prefix if it exists
        model_id = model_id[len("hf-hub:") :] if model_id.startswith("hf-hub:") else model_id
        ckpt_path = download_pretrained_from_hf(
            model_id, cache_dir=os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/"))
        )
        return ckpt_path

    @torch.inference_mode()
    @torch.cuda.amp.autocast(enabled=True)
    def __call__(self, list_of_texts: List[str], normalize: bool = True) -> torch.Tensor:
        text_tokens = self.tokenizer(list_of_texts, context_length=128).to(self.device)
        embeddings = self.model.encode_text(text_tokens)
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings


class Siglip2TextEncoder(CLIPTextEncoderInterace):
    CHANNEL_DIM = 1152

    def __init__(
        self,
        model_id: str = "google/siglip2-so400m-patch16-384",
        device: str = "cuda",
        attn_implementation: str = "flash_attention_2",
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        # Disable tokenizer parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.model.vision_model = None  # Remove vision model
        self.device = device

    @torch.inference_mode()
    @torch.cuda.amp.autocast(enabled=True)
    def __call__(self, list_of_texts: List[str], normalize: bool = True) -> torch.Tensor:
        # Length is 64 https://huggingface.co/docs/transformers/main/model_doc/siglip2
        text_inputs = self.tokenizer(
            list_of_texts,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model.get_text_features(**text_inputs)
        if normalize:
            outputs = torch.nn.functional.normalize(outputs, dim=-1)
        return outputs


if __name__ == "__main__":
    texts = [
        "a very long text that is a sentence but has a lot of words, details, and information, and very very very long",
        "a photo of a cat",
    ]

    # Siglip2
    siglip_encoder = Siglip2TextEncoder()
    print("Siglip2 embeddings:", siglip_encoder(texts))

    # OpenCLIP
    openclip_encoder = OpenCLIPTextEncoder(model_id="hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP")
    print("OpenCLIP embeddings:", openclip_encoder(texts))
