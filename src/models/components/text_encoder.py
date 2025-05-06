from typing import List

import abc

import torch
from transformers import AutoTokenizer, AutoModel
from open_clip import create_model_from_pretrained, get_tokenizer


class CLIPTextEncoderInterace(abc.ABC):
    model: torch.nn.Module
    CHANNEL_DIM: int

    def freeze_encoder(self):
        for params in self.model.parameters():
            params.requires_grad = False

    @abc.abstractmethod
    def __call__(self, list_of_texts: List[str]) -> torch.Tensor:
        pass


def get_text_encoder(
    model_type: str,
    device: str,
    **kwargs,
) -> CLIPTextEncoderInterace:
    if model_type == "siglip":
        return Siglip2TextEncoder(device=device, **kwargs)
    elif model_type == "recap":
        return RecapCLIPTextEncoder(device=device, **kwargs)
    else:
        raise ValueError(f"Model type {model_type} not supported")


class Siglip2TextEncoder(CLIPTextEncoderInterace):
    CHANNEL_DIM = 1152

    def __init__(
        self,
        model_id: str = "google/siglip2-so400m-patch16-384",
        device: str = "cuda",
        attn_implementation: str = "flash_attention_2",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.model.vision_model = None  # Remove vision model
        self.device = device
        self.freeze_encoder()

    @torch.inference_mode()
    @torch.cuda.amp.autocast(enabled=True)
    def __call__(self, list_of_texts: List[str]) -> torch.Tensor:
        # Length is 64 https://huggingface.co/docs/transformers/main/model_doc/siglip2
        text_inputs = self.tokenizer(
            list_of_texts,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model.get_text_features(**text_inputs)
        return outputs


class RecapCLIPTextEncoder(CLIPTextEncoderInterace):
    CHANNEL_DIM = 768

    def __init__(
        self,
        model_id: str = "hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.tokenizer = get_tokenizer(model_id)
        precision = {torch.float16: "fp16", torch.bfloat16: "bf16"}[torch_dtype]
        self.model, _ = create_model_from_pretrained(
            model_id,
            device=device,
            precision=precision,
        )
        self.device = device
        self.freeze_encoder()

    @torch.inference_mode()
    @torch.cuda.amp.autocast(enabled=True)
    def __call__(self, list_of_texts: List[str]) -> torch.Tensor:
        text_tokens = self.tokenizer(list_of_texts, context_length=128).to(self.device)
        return self.model.encode_text(text_tokens)


if __name__ == "__main__":
    texts = [
        "a very long text that is a sentence but has a lot of words, details, and information, and very very very long",
        "a photo of a cat",
    ]

    # Siglip2
    siglip_encoder = Siglip2TextEncoder()
    print("Siglip2 embeddings:", siglip_encoder(texts))

    # Recap CLIP
    recap_encoder = RecapCLIPTextEncoder()
    print("RecapCLIP embeddings:", recap_encoder(texts))
