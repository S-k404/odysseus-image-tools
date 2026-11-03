"""Detect and inpaint one image or a folder; shared by the command line tool and the MCP server."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageOps

from .detect import Region, detect_all, grow_mask, manual_mask, preview
from .inpaint import Inpainter

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
OUT_DIR = Path(os.environ.get("UNCENSOR_OUT_DIR") or Path.home() / "Pictures" / "odysseus-uncensor")


def load_rgb(path: Path) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        base = Image.new("RGBA", img.size, (255, 255, 255, 255))
        base.alpha_composite(img)
        img = base
    return np.array(img.convert("RGB"))


def check_image_path(text: str) -> Path:
    path = Path(text).expanduser()
    if not path.is_file():
        raise ValueError(f"no such file: {text}")
    if path.suffix.lower() not in IMAGE_EXTS:
        raise ValueError(f"not an image type I handle: {path.suffix}")
    return path


def parse_boxes(text: str) -> list[tuple[int, int, int, int]]:
    """"x,y,w,h;x,y,w,h" -> boxes."""
    boxes = []
    for chunk in text.replace("\n", ";").split(";"):
        if chunk.strip():
            x, y, w, h = (int(float(v)) for v in chunk.replace(" ", "").split(","))
            boxes.append((x, y, w, h))
    return boxes


def free_name(directory: Path, stem: str, suffix: str) -> Path:
    """A path that does not exist yet, so nothing is ever overwritten."""
    candidate = directory / f"{stem}{suffix}"
    n = 2
    while candidate.exists():
        candidate = directory / f"{stem}_{n}{suffix}"
        n += 1
    return candidate


def find_regions(path: Path, kinds=("mosaic", "bars"), boxes=None, mask_path: Path | None = None):
    """(rgb, regions, mask) from a manual mask file, manual boxes, or automatic detection."""
    rgb = load_rgb(path)
    if mask_path is not None:
        m = np.array(Image.open(mask_path).convert("L").resize((rgb.shape[1], rgb.shape[0])))
        mask = (m > 127).astype(np.uint8) * 255
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        regions = [Region(int(s[0]), int(s[1]), int(s[2]), int(s[3]), "manual") for s in stats[1:]]
    elif boxes:
        regions, mask = manual_mask(rgb.shape[:2], boxes)
    else:
        regions, mask = detect_all(rgb, tuple(kinds))
    return rgb, regions, mask


def describe(regions: list[Region]) -> list[str]:
    return [f"{i}. {r.kind} at x={r.x} y={r.y} size {r.w}x{r.h}" + (f" block {r.block}px" if r.block else "")
            for i, r in enumerate(regions, 1)]


def write_preview(path: Path, rgb, regions, mask, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = free_name(out_dir, f"{path.stem}_regions", ".png")
    Image.fromarray(preview(rgb, regions, mask)).save(target)
    return target


def uncensor(path: Path, inpainter: Inpainter, *, out_dir: Path = OUT_DIR, kinds=("mosaic", "bars"), boxes=None,
             mask_path: Path | None = None, prompt: str = "", negative: str = "", steps: int = 30,
             guidance: float = 7.5, strength: float = 0.99, grow: int | None = None, seed: int | None = None,
             max_regions: int = 12, progress: Callable[[str], None] | None = None) -> dict:
    """Detect (or take) the censored regions, inpaint them, write `<stem>_uncensored.png`. The original is untouched."""
    started = time.time()
    rgb, regions, mask = find_regions(path, kinds, boxes, mask_path)
    if not regions:
        return {"output": None, "regions": [], "note": "nothing found to fill in"}
    if len(regions) > max_regions:
        return {"output": None, "regions": describe(regions),
                "note": f"{len(regions)} regions found, more than max_regions={max_regions}; "
                        "pass regions yourself or raise max_regions"}
    pad = grow if grow is not None else max([r.block or 8 for r in regions]) // 2 + 4
    out = inpainter.inpaint(rgb, grow_mask(mask, pad), prompt, negative, steps, guidance, strength, seed,
                            progress=progress)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = free_name(out_dir, f"{path.stem}_uncensored", ".png")
    Image.fromarray(out).save(target)
    return {"output": str(target), "regions": describe(regions), "seconds": round(time.time() - started, 1)}


def list_images(folder: Path) -> list[Path]:
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()),
                  key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name.lower())])
