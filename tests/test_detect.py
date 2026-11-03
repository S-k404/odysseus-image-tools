"""Check the censor detector on harmless synthetic data with known ground truth.

Run: .venv/bin/python tests/test_detect.py
Takes a picture, pixelates known rectangles (different block sizes, offsets, JPEG noise), and checks
that the detector finds them (IoU of the mask) and finds nothing on pictures without any.
"""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uncensor.detect import detect_all, detect_bars, detect_mosaic  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def pixelate(img: np.ndarray, box, block: int) -> np.ndarray:
    """Block-average the box with the grid anchored at the box corner, like an editor's mosaic filter."""
    x0, y0, x1, y1 = box
    out = img.copy()
    for y in range(y0, y1, block):
        for x in range(x0, x1, block):
            ye, xe = min(y + block, y1), min(x + block, x1)
            out[y:ye, x:xe] = img[y:ye, x:xe].reshape(-1, 3).mean(axis=0).astype(np.uint8)
    return out


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=quality)
    return np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def procedural(w=900, h=1200, seed=1, panel=(30, 30, 30)) -> np.ndarray:
    """Flat panels, gradients, lines and text: things that must NOT look like a mosaic."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), (250, 250, 250))
    d = ImageDraw.Draw(img)
    for i in range(0, h, 300):
        d.rectangle([20, i + 20, w - 20, i + 280], outline=(0, 0, 0), width=4)
        d.rectangle([60, i + 60, 300, i + 200], fill=panel)
        for k in range(8):
            d.text((340, i + 50 + k * 22), "The quick brown fox jumps over the lazy dog", fill=(0, 0, 0))
    arr = np.array(img)
    grad = np.linspace(0, 255, w, dtype=np.uint8)[None, :, None].repeat(160, 0).repeat(3, 2)
    arr[900:1060, :, :] = grad
    arr = np.clip(arr.astype(np.int16) + rng.integers(-2, 3, arr.shape), 0, 255).astype(np.uint8)
    return arr


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def truth_mask(shape, boxes):
    m = np.zeros(shape[:2], bool)
    for x0, y0, x1, y1 in boxes:
        m[y0:y1, x0:x1] = True
    return m


failures = []


def check(name: str, ok: bool, detail: str):
    print(("PASS " if ok else "FAIL ") + name + "  " + detail)
    if not ok:
        failures.append(name)


photo = np.array(Image.open(DATA / "lighthouse512.png").convert("RGB"))
big = np.array(Image.open(DATA / "lighthouse_small.png").convert("RGB").resize((1000, 1000)))

# 1. Mosaics: several block sizes, boxes not aligned to any global grid, clean and JPEG.
cases = [
    ("photo b8", photo, [(100, 90, 300, 210)], 8, None),
    ("photo b12", photo, [(60, 140, 260, 300)], 12, None),
    ("photo b6 small", photo, [(200, 60, 320, 140)], 6, None),
    ("big b16", big, [(130, 220, 430, 400)], 16, None),
    ("big b10 jpeg90", big, [(500, 300, 780, 520)], 10, 90),
    ("big two boxes b9+b14", big, [(80, 80, 260, 200), (560, 640, 860, 860)], None, None),
]
for name, base, boxes, block, q in cases:
    img = base
    sizes = [block] * len(boxes) if block else [9, 14]
    for box, b in zip(boxes, sizes):
        img = pixelate(img, box, b)
    if q:
        img = jpeg(img, q)
    regions, mask = detect_mosaic(img)
    score = iou(mask > 0, truth_mask(img.shape, boxes))
    check(f"mosaic {name}", score >= 0.8, f"IoU={score:.2f} regions={[(r.x, r.y, r.w, r.h, r.block) for r in regions]}")

# 2. Nothing to find.
for name, img in [("photo", photo), ("big photo", big), ("procedural page", procedural())]:
    regions, mask = detect_all(img)
    check(f"clean {name}", len(regions) == 0, f"regions={[(r.kind, r.x, r.y, r.w, r.h) for r in regions]}")

# 3. Black bars on a photo and on the page; thin lines and the dark panel on the page are not bars.
bar = photo.copy()
bar[200:240, 80:300] = 0
regions, mask = detect_bars(bar)
score = iou(mask > 0, truth_mask(bar.shape, [(80, 200, 300, 240)]))
check("bar on photo", score >= 0.9 and len(regions) == 1, f"IoU={score:.2f} n={len(regions)}")
page = procedural(panel=(0, 0, 0))
regions, _ = detect_bars(page)
check("bars on page: 4 black panels found, gradient and frames ignored", len(regions) == 4, f"n={len(regions)} {[(r.x, r.y, r.w, r.h) for r in regions]}")

print("\nFAILED: " + ", ".join(failures) if failures else "\nall passed")
sys.exit(1 if failures else 0)
