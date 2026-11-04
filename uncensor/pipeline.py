"""Detect and inpaint one image or a folder; shared by the command line tool and the MCP server."""
from __future__ import annotations

import io
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

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".png_", ".rpgmvp", ".rpgmvo"}
OUT_DIR = Path(os.environ.get("UNCENSOR_OUT_DIR") or Path.home() / "Downloads" / "odysseus-decensor")

PNG_MAGIC = bytes([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52])
RPGMV_HEADER = b"RPGMV\x00\x00\x00\x00\x03\x01\x00\x00\x00\x00\x00"
RPGMZ_HEADER = b"RPGMZ\x00\x00\x00\x00\x03\x01\x00\x00\x00\x00\x00"


def load_rgb(path: Path | str) -> np.ndarray:
    raw = Path(path).read_bytes()
    if raw.startswith(RPGMV_HEADER) or raw.startswith(RPGMZ_HEADER):
        # RPG Maker MV/MZ encrypted image: first 16 bytes are RPGMV/MZ header,
        # next 16 bytes are XOR encrypted PNG header (which is always PNG_MAGIC).
        raw = PNG_MAGIC + raw[32:]
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        base = Image.new("RGBA", img.size, (255, 255, 255, 255))
        base.alpha_composite(img)
        img = base
    return np.array(img.convert("RGB"))


def check_image_path(text: str) -> Path:
    raw = str(text or "").strip()
    if raw:
        p = Path(raw).expanduser()
        if p.is_file():
            if p.suffix.lower() not in IMAGE_EXTS:
                raise ValueError(f"not an image type I handle: {p.suffix}")
            return p

    # Check Odysseus uploads manifest if available
    odysseus_uploads = Path(__file__).resolve().parent.parent.parent / "Odysseus" / "data" / "uploads"
    uploads_json = odysseus_uploads / "uploads.json"
    if uploads_json.is_file():
        try:
            import json
            meta = json.loads(uploads_json.read_text())
            clean_q = Path(raw).name.lower() if raw else ""
            generic_terms = {"it", "this", "image", "the image", "attached", "latest", "file", "photo", "pic", "the picture", ""}
            if not raw or clean_q in generic_terms:
                candidates = []
                for v in meta.values():
                    p_str = v.get("path") or ""
                    if p_str.startswith("/app/data/uploads/"):
                        p_str = str(odysseus_uploads / p_str[len("/app/data/uploads/"):])
                    lp = Path(p_str)
                    if lp.is_file() and lp.suffix.lower() in IMAGE_EXTS:
                        candidates.append((lp.stat().st_mtime, lp))
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    return candidates[0][1]
            else:
                matches = []
                for v in meta.values():
                    name = str(v.get("name") or "").lower()
                    uid = str(v.get("id") or "").lower()
                    if clean_q in (name, uid) or Path(clean_q).stem in (Path(name).stem, Path(uid).stem):
                        p_str = v.get("path") or ""
                        if p_str.startswith("/app/data/uploads/"):
                            p_str = str(odysseus_uploads / p_str[len("/app/data/uploads/"):])
                        lp = Path(p_str)
                        if lp.is_file() and lp.suffix.lower() in IMAGE_EXTS:
                            uploaded_time = str(v.get("uploaded_at") or v.get("created_at") or "")
                            mtime = lp.stat().st_mtime
                            matches.append((uploaded_time, mtime, lp))
                if matches:
                    matches.sort(key=lambda x: (x[0], x[1]), reverse=True)
                    return matches[0][2]
        except Exception:
            pass

    # Candidate directories to resolve uploaded files or relative names
    candidate_dirs = [
        odysseus_uploads,
        Path.home() / "Downloads",
        Path.home() / "Downloads" / "odysseus-decensor",
        Path.home() / "Pictures",
        Path.home() / "Pictures" / "odysseus-uncensor",
        Path.home() / "Documents" / "TEST" / "pictures",
        Path.cwd(),
    ]

    generic_terms = {"it", "this", "image", "the image", "attached", "latest", "file", "photo", "pic", "the picture", ""}
    low = raw.lower()
    if low in generic_terms or not raw:
        for cdir in candidate_dirs:
            if cdir.is_dir():
                found_images = [
                    f for f in cdir.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS and not f.name.startswith(".")
                ]
                if found_images:
                    found_images.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    return found_images[0]

    clean_name = Path(raw).name
    for cdir in candidate_dirs:
        if not cdir.is_dir():
            continue
        direct = cdir / clean_name
        if direct.is_file() and direct.suffix.lower() in IMAGE_EXTS:
            return direct
        for f in cdir.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                if clean_name.lower() in f.name.lower() or f.stem.lower() == Path(clean_name).stem.lower():
                    return f

    if raw:
        p = Path(raw).expanduser()
        if not p.is_file():
            raise ValueError(f"no such file: {text}")
        if p.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"not an image type I handle: {p.suffix}")
        return p

    raise ValueError("no image path provided and no recent upload found")


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


def find_regions(path: Path, kinds=("mosaic", "bars", "spots"), boxes=None, mask_path: Path | None = None):
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


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


def uncensor(path: Path, inpainter: Inpainter, *, out_dir: Path = OUT_DIR, kinds=("mosaic", "bars", "spots"), boxes=None,
             mask_path: Path | None = None, prompt: str = "", negative: str = "", steps: int = 30,
             guidance: float = 7.5, strength: float = 0.99, grow: int | None = None, seed: int | None = None,
             max_regions: int = 48, model: str | Path | None = None, style: str = "auto", anatomy: str = "auto",
             progress: Callable[[str], None] | None = None) -> dict:
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
    out = inpainter.inpaint(rgb, grow_mask(mask, pad), prompt=prompt, negative=negative, steps=steps,
                            guidance=guidance, strength=strength, seed=seed,
                            model=model, style=style, anatomy=anatomy, progress=progress)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = free_name(out_dir, f"{path.stem}_uncensored", ".png")
    Image.fromarray(out).save(target)
    return {"output": str(target), "regions": describe(regions), "seconds": round(time.time() - started, 1)}


def uncensor_video(path: Path, inpainter: Inpainter, *, out_dir: Path = OUT_DIR, kinds=("mosaic", "bars", "spots"),
                   prompt: str = "", negative: str = "", steps: int = 20, guidance: float = 7.5,
                   strength: float = 0.99, model: str | Path | None = None, style: str = "auto",
                   anatomy: str = "auto", progress: Callable[[str], None] | None = None) -> dict:
    """Decensor anime/manga video file frame by frame and preserve original audio track using ffmpeg."""
    import subprocess
    import tempfile

    started = time.time()
    say = progress or (lambda _msg: None)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    say(f"opening video: {path.name} ({width}x{height}, {fps:.2f} fps, {total_frames} frames)")

    out_dir.mkdir(parents=True, exist_ok=True)
    target = free_name(out_dir, f"{path.stem}_uncensored", path.suffix)

    with tempfile.TemporaryDirectory(prefix="uncensor_vid_") as tmpdir:
        tmp_video = Path(tmpdir) / f"raw_frames{path.suffix}"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, (width, height))

        frame_idx = 0
        censored_frames = 0
        last_mask = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            regions, mask = detect_all(rgb, tuple(kinds))
            if regions:
                censored_frames += 1
                last_mask = mask
                pad = max([r.block or 8 for r in regions]) // 2 + 4
                inpainted_rgb = inpainter.inpaint(rgb, grow_mask(mask, pad), prompt=prompt, negative=negative,
                                                  steps=steps, guidance=guidance, strength=strength,
                                                  model=model, style=style, anatomy=anatomy)
                frame = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)
            elif last_mask is not None and frame_idx % 2 != 0:
                # Slight temporal cohesion check
                last_mask = None
            writer.write(frame)
            if frame_idx % 30 == 0:
                say(f"processed {frame_idx}/{total_frames} frames ({censored_frames} censored)")

        cap.release()
        writer.release()

        # Remux original audio if present
        say("remuxing video and preserving original audio track...")
        cmd = ["ffmpeg", "-y", "-i", str(tmp_video), "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-c:a", "copy", "-map", "0:v:0", "-map", "1:a:0?", "-shortest", str(target)]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            # Fallback without audio map if no audio stream exists
            cmd_fallback = ["ffmpeg", "-y", "-i", str(tmp_video), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target)]
            subprocess.run(cmd_fallback, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return {"output": str(target), "total_frames": total_frames, "censored_frames": censored_frames,
            "seconds": round(time.time() - started, 1)}


def list_images(folder: Path, recursive: bool = True) -> list[Path]:
    all_exts = IMAGE_EXTS | VIDEO_EXTS
    candidates = [p for p in folder.iterdir() if p.suffix.lower() in all_exts and p.is_file()]
    if not candidates:
        pics = folder / "pictures"
        if pics.is_dir():
            candidates = [p for p in pics.iterdir() if p.suffix.lower() in all_exts and p.is_file()]
    if not candidates and recursive:
        candidates = [p for p in folder.rglob("*") if p.suffix.lower() in all_exts and p.is_file()]
    return sorted(candidates,
                  key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name.lower())])
