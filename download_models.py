"""Download the inpainting model used by the uncensor tool (safetensors only, fp16)."""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
DEST = Path(__file__).resolve().parent / "models" / "sdxl-inpainting"

path = snapshot_download(
    REPO,
    local_dir=DEST,
    allow_patterns=[
        "model_index.json", "scheduler/*.json", "tokenizer*/*", "*/config.json",
        "text_encoder/model.fp16.safetensors", "text_encoder_2/model.fp16.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/diffusion_pytorch_model.safetensors",  # fp32: the SDXL VAE overflows in fp16
    ],
)
print("downloaded to", path)
