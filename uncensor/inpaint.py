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

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "sdxl-inpainting"
POSITIVE = "high quality, detailed, consistent with the surrounding image, matching its style, lighting and colours"
NEGATIVE = "mosaic, pixelated, blocky, censored, censor bar, blur, low quality, jpeg artifacts, watermark, text"
NATIVE = 1024  # SDXL's trained size: the long side of every crop is scaled to this


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
        self.model_dir = Path(os.environ.get("UNCENSOR_MODEL_DIR") or model_dir or DEFAULT_MODEL)
        self.idle_seconds = idle_seconds
        self._pipe = None
        self._lock = threading.RLock()
        self._last_used = 0.0

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    @property
    def busy(self) -> bool:
        if self._lock.acquire(blocking=False):
            self._lock.release()
            return False
        return True

    def _load(self):
        import torch
        from diffusers import StableDiffusionXLInpaintPipeline

        if not (self.model_dir / "model_index.json").exists():
            raise FileNotFoundError(f"model not found in {self.model_dir}; run download_models.py")
        on_gpu = torch.backends.mps.is_available()
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float16 if on_gpu else torch.float32,
            variant="fp16" if on_gpu else None,
            use_safetensors=True,
        )
        pipe.to("mps" if on_gpu else "cpu")
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def unload(self) -> bool:
        with self._lock:
            if self._pipe is None:
                return False
            self._pipe = None
            gc.collect()
            try:
                import torch
                torch.mps.empty_cache()
            except Exception:
                pass
            return True

    def unload_if_idle(self) -> bool:
        if self._pipe is not None and not self.busy and time.time() - self._last_used > self.idle_seconds:
            return self.unload()
        return False

    def inpaint(self, image: np.ndarray, mask: np.ndarray, prompt: str = "", negative: str = "", steps: int = 30,
                guidance: float = 7.5, strength: float = 0.99, seed: int | None = None, feather: float = 3.0,
                progress: Callable[[str], None] | None = None) -> np.ndarray:
        """image (H, W, 3) uint8 RGB, mask (H, W) uint8 (255 = replace). Returns a new image."""
        import torch
        from PIL import Image

        say = progress or (lambda _msg: None)
        with self._lock:
            if self._pipe is None:
                say("loading the inpainting model")
                self._pipe = self._load()
            result = image.copy()
            parts = components(mask)
            for n, (bbox, part) in enumerate(parts, 1):
                x0, y0, x1, y1 = window_for(bbox, mask.shape)
                crop = Image.fromarray(result[y0:y1, x0:x1])
                m = Image.fromarray(part[y0:y1, x0:x1])
                scale = NATIVE / max(crop.width, crop.height)
                size = (_round8(crop.width * scale), _round8(crop.height * scale))
                say(f"region {n}/{len(parts)} at {bbox[0]},{bbox[1]} ({bbox[2]}x{bbox[3]}) -> {size[0]}x{size[1]}")
                generator = torch.Generator("cpu").manual_seed(seed if seed is not None else int(time.time()) % 2**31)
                out = self._pipe(
                    prompt=f"{prompt}, {POSITIVE}" if prompt else POSITIVE,
                    negative_prompt=f"{negative}, {NEGATIVE}" if negative else NEGATIVE,
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
