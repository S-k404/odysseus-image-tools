"""Fill masked regions with SDXL inpainting on the Mac's GPU (MPS).

Each connected region is cut out with surrounding context, scaled to the model's native size,
inpainted, scaled back and blended in with a soft edge, so pixels outside the mask are left as they were.
The weights are loaded on first use and dropped after `idle_seconds` without a request.
"""
from __future__ import annotations

import gc
import os
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LAMA_MODEL = MODELS_DIR / "anime-manga-big-lama.pt"
ANIME_MODEL = MODELS_DIR / "animagine-xl-inpainting"
GENERAL_MODEL = MODELS_DIR / "sdxl-inpainting"
DEFAULT_MODEL = LAMA_MODEL if LAMA_MODEL.exists() else (ANIME_MODEL if (ANIME_MODEL / "model_index.json").exists() else GENERAL_MODEL)

POSITIVE_GENERAL = "high quality, detailed, consistent with the surrounding image, matching its style, lighting and colours"
NEGATIVE_GENERAL = "mosaic, pixelated, blocky, censored, censor bar, blur, low quality, jpeg artifacts, watermark, text"

POSITIVE_ANIME = "masterpiece, best quality, anime style, manga art, clean lineart, flawless anatomy, perfectly matching surrounding art style and color palette"
POSITIVE_ANIME_FEMALE = "masterpiece, best quality, anime style, manga art, clean lineart, uncensored, pussy, clitoris, labia, vaginal slit, detailed anatomy, flawless skin"
POSITIVE_ANIME_MALE = "masterpiece, best quality, anime style, manga art, clean lineart, uncensored, penis, glans, erection, veins, shaft, foreskin, detailed male anatomy"
NEGATIVE_ANIME = "mosaic, pixelated, blocky, censored, censor bar, blur, low quality, worst quality, jpeg artifacts, realistic, photorealistic, 3d, photo, render, western comic, watermark, text"

NATIVE = 1024  # SDXL's trained size: the long side of every crop is scaled to this


def resolve_model_dir(model: str | Path | None = None) -> Path:
    if model is None:
        env = os.environ.get("UNCENSOR_MODEL_DIR") or os.environ.get("UNCENSOR_MODEL")
        if env:
            return Path(env)
        if LAMA_MODEL.exists():
            return LAMA_MODEL
        if (ANIME_MODEL / "model_index.json").exists():
            return ANIME_MODEL
        return GENERAL_MODEL

    if isinstance(model, str):
        low = model.strip().lower()
        if low in ("lama", "manga", "anime-manga", "fast", "er0manga"):
            return LAMA_MODEL
        if low in ("anime", "animagine"):
            return LAMA_MODEL if LAMA_MODEL.exists() else ANIME_MODEL
        if low in ("general", "sdxl", "realistic"):
            return GENERAL_MODEL
    p = Path(model)
    if not p.is_absolute() and (MODELS_DIR / p).exists():
        return MODELS_DIR / p
    return p


def _round8(v: float) -> int:
    return max(64, int(round(v / 8)) * 8)


def components(mask: np.ndarray, merge_gap: int = 24):
    """Regions to inpaint one by one: (bbox, mask) per group of mask pieces closer than `merge_gap` pixels."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * merge_gap + 1, 2 * merge_gap + 1))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(mask, k), connectivity=8)
    out = []
    for i in range(1, count):
        part = np.where(labels == i, mask, 0).astype(np.uint8)
        if part.any():
            x, y, w, h = (int(v) for v in stats[i, :4])
            out.append(((x, y, w, h), part))
    return out


def window_for(bbox, shape, context: float = 0.75, min_side: int = 512):
    """Crop rectangle (x0, y0, x1, y1) around bbox: square if the image allows, with `context` of surroundings."""
    x, y, w, h = bbox
    height, width = shape
    side = max(int(max(w, h) * (1 + 2 * context)), min_side)
    ww, wh = min(side, width), min(side, height)
    x0 = int(np.clip(x + w / 2 - ww / 2, 0, width - ww))
    y0 = int(np.clip(y + h / 2 - wh / 2, 0, height - wh))
    return x0, y0, x0 + ww, y0 + wh


class Inpainter:
    def __init__(self, model_dir: str | Path | None = None, idle_seconds: float = 300):
        self.model_dir = resolve_model_dir(model_dir)
        self.idle_seconds = idle_seconds
        self._pipe = None
        self._model = None
        self._lock = threading.RLock()
        self._last_used = 0.0

    @property
    def is_lama(self) -> bool:
        return self.model_dir.suffix.lower() == ".pt" or "lama" in str(self.model_dir).lower()

    @property
    def loaded(self) -> bool:
        return self._pipe is not None or self._model is not None

    @property
    def busy(self) -> bool:
        if self._lock.acquire(blocking=False):
            self._lock.release()
            return False
        return True

    def set_model(self, model: str | Path | None):
        target = resolve_model_dir(model)
        if target != self.model_dir:
            self.unload()
            self.model_dir = target

    def _load(self):
        import torch
        if self.is_lama:
            if not self.model_dir.exists():
                raise FileNotFoundError(f"LaMa model not found at {self.model_dir}")
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = torch.jit.load(str(self.model_dir), map_location=device)
            model.eval()
            return model

        from diffusers import StableDiffusionXLInpaintPipeline

        if not (self.model_dir / "model_index.json").exists():
            raise FileNotFoundError(f"model not found in {self.model_dir}; run download script")
        on_gpu = torch.backends.mps.is_available()
        has_fp16 = bool(list(self.model_dir.glob("unet/*.fp16.safetensors")))
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float16 if on_gpu else torch.float32,
            variant="fp16" if (on_gpu and has_fp16) else None,
            use_safetensors=True,
        )
        pipe.to("mps" if on_gpu else "cpu")
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def unload(self) -> bool:
        with self._lock:
            unloaded = False
            if self._pipe is not None:
                self._pipe = None
                unloaded = True
            if self._model is not None:
                self._model = None
                unloaded = True
            if unloaded:
                gc.collect()
                try:
                    import torch
                    torch.mps.empty_cache()
                except Exception:
                    pass
            return unloaded

    def unload_if_idle(self) -> bool:
        if (self._pipe is not None or self._model is not None) and not self.busy and time.time() - self._last_used > self.idle_seconds:
            return self.unload()
        return False

    def _inpaint_lama(self, image: np.ndarray, mask: np.ndarray, feather: float = 3.0) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        h, w = image.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8

        img_t = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)

        if pad_h > 0 or pad_w > 0:
            img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode="reflect")
            mask_t = F.pad(mask_t, (0, pad_w, 0, pad_h), mode="replicate")

        img_t = img_t * (1.0 - mask_t)

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        with torch.no_grad():
            out_t = self._model(img_t.to(device), mask_t.to(device))

        out_np = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if pad_h > 0 or pad_w > 0:
            out_np = out_np[:h, :w]
        out_img = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)

        soft = mask.astype(np.float32) / 255.0
        alpha = np.maximum(soft, cv2.GaussianBlur(soft, (0, 0), feather))[..., None]
        return (out_img.astype(np.float32) * alpha + image.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

    def inpaint(self, image: np.ndarray, mask: np.ndarray, prompt: str = "", negative: str = "", steps: int = 30,
                guidance: float = 7.5, strength: float = 0.99, seed: int | None = None, feather: float = 3.0,
                model: str | Path | None = None, style: str = "auto", anatomy: str = "auto",
                progress: Callable[[str], None] | None = None) -> np.ndarray:
        """image (H, W, 3) uint8 RGB, mask (H, W) uint8 (255 = replace). Returns a new image."""
        import torch
        from PIL import Image

        say = progress or (lambda _msg: None)
        with self._lock:
            if model is not None:
                self.set_model(model)

            if self.is_lama:
                if self._model is None:
                    say(f"loading fast manga inpainting model: {self.model_dir.name}")
                    self._model = self._load()
                say(f"inpainting with anime-manga LaMa ({image.shape[1]}x{image.shape[0]})")
                result = self._inpaint_lama(image, mask, feather=feather)
                self._last_used = time.time()
                return result

            if self._pipe is None:
                say(f"loading inpainting model: {self.model_dir.name}")
                self._pipe = self._load()

            is_anime = (style.lower() in ("anime", "manga")) or (
                style.lower() == "auto" and any(k in str(self.model_dir).lower() for k in ("anime", "animagine", "manga"))
            )
            base_neg = NEGATIVE_ANIME if is_anime else NEGATIVE_GENERAL
            full_negative = f"{negative}, {base_neg}" if negative else base_neg

            # Pre-demosaic the masked area to remove blocky grid lattice frequencies from the latent base
            demosaiced = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA) if mask.any() else image
            result = image.copy()
            parts = components(mask)
            for n, (bbox, part) in enumerate(parts, 1):
                x0, y0, x1, y1 = window_for(bbox, mask.shape)
                crop = Image.fromarray(demosaiced[y0:y1, x0:x1])
                m = Image.fromarray(part[y0:y1, x0:x1])
                scale = NATIVE / max(crop.width, crop.height)
                size = (_round8(crop.width * scale), _round8(crop.height * scale))

                # Resolve anime prompt conditioning based on anatomy
                if is_anime:
                    anat_mode = anatomy.lower()
                    if anat_mode == "female":
                        base_pos = POSITIVE_ANIME_FEMALE
                    elif anat_mode == "male":
                        base_pos = POSITIVE_ANIME_MALE
                    elif anat_mode == "auto":
                        # Elongated / horizontal or large area -> male; vertical / slit -> female
                        bw, bh = bbox[2], bbox[3]
                        if bw > 1.3 * bh or (bw * bh > 12000):
                            base_pos = POSITIVE_ANIME_MALE
                        else:
                            base_pos = POSITIVE_ANIME_FEMALE
                    else:
                        base_pos = POSITIVE_ANIME
                else:
                    base_pos = POSITIVE_GENERAL

                region_prompt = f"{prompt}, {base_pos}" if prompt else base_pos
                say(f"region {n}/{len(parts)} at {bbox[0]},{bbox[1]} ({bbox[2]}x{bbox[3]}) -> {size[0]}x{size[1]}")
                generator = torch.Generator("cpu").manual_seed(seed if seed is not None else int(time.time()) % 2**31)
                out = self._pipe(
                    prompt=region_prompt,
                    negative_prompt=full_negative,
                    image=crop.resize(size, Image.LANCZOS),
                    mask_image=m.resize(size, Image.NEAREST),
                    height=size[1], width=size[0],
                    num_inference_steps=steps, guidance_scale=guidance, strength=strength,
                    generator=generator,
                ).images[0]
                filled = np.array(out.resize(crop.size, Image.LANCZOS))
                # Inside the mask take the new pixels; fade out over a few pixels so there is no visible seam.
                soft = part[y0:y1, x0:x1].astype(np.float32) / 255
                alpha = np.maximum(soft, cv2.GaussianBlur(soft, (0, 0), feather))[..., None]
                result[y0:y1, x0:x1] = (filled * alpha + result[y0:y1, x0:x1] * (1 - alpha)).astype(np.uint8)
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
            self._last_used = time.time()
            return result
