"""Download the inpainting models used by the uncensor tool.
Default: specialized anime-manga Big-LaMa model (~196 MB, fast, seamless lineart).
Optional: SDXL inpainting (6.6 GB) via --sdxl flag.
"""
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LAMA_URL = "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt"
LAMA_DEST = MODELS_DIR / "anime-manga-big-lama.pt"

if not LAMA_DEST.exists():
    print(f"Downloading anime-manga LaMa inpainting model to {LAMA_DEST}...")
    urllib.request.urlretrieve(LAMA_URL, LAMA_DEST)
    print("Downloaded anime-manga LaMa model!")
else:
    print(f"Anime-manga LaMa model already present at {LAMA_DEST}")

if "--sdxl" in sys.argv:
    from huggingface_hub import snapshot_download

    REPO = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    DEST = MODELS_DIR / "sdxl-inpainting"
    path = snapshot_download(
        REPO,
        local_dir=DEST,
        allow_patterns=[
            "model_index.json", "scheduler/*.json", "tokenizer*/*", "*/config.json",
            "text_encoder/model.fp16.safetensors", "text_encoder_2/model.fp16.safetensors",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ],
    )
    print("SDXL inpainting downloaded to", path)
