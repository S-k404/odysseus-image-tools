"""Find mosaic (pixelation) and solid-bar censoring in an image. No ML: numpy + OpenCV.

A mosaic is a grid of flat blocks, so its edges fall only on a regular lattice (every `b` pixels)
and each block is almost one colour. The detector looks for that in tiles, then confirms block by
block. Solid black bars are found as large, very rectangular, near-black components.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int
    kind: str  # "mosaic" | "bar" | "manual"
    block: int | None = None  # mosaic block size in pixels

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ mosaic

def _lattice(profile: np.ndarray, tile: int, lo: int, hi: int, share_min: float, occ_min: float):
    """For each tile's edge profile, the block size and phase whose lattice holds the edges.

    profile: (tiles, n) edge counts on the boundary after pixel i. Returns (block, phase, ok), each (tiles,).
    A lattice only counts when (almost) all edge energy sits on it AND most lattice lines carry edges;
    the second test rejects divisors of the true block size, whose lattices are mostly empty.
    """
    nt, n = profile.shape
    total = profile.sum(axis=1)
    rows = np.arange(nt)
    best_score = np.zeros(nt)
    best_b = np.zeros(nt, dtype=int)
    best_p = np.zeros(nt, dtype=int)
    for b in range(lo, hi + 1):
        pad = (-n) % b
        m = np.pad(profile, ((0, 0), (0, pad))).reshape(nt, -1, b)
        sums = m.sum(axis=1)
        p = sums.argmax(axis=1)
        p_prev = (p - 1) % b
        p_next = (p + 1) % b
        adj = np.maximum(sums[rows, p_prev], sums[rows, p_next])
        share = (sums[rows, p] + adj) / np.maximum(total, 1)
        lines = m[rows, :, p]  # (nt, cycles)
        count = (n - p + b - 1) // b
        valid = np.arange(lines.shape[1])[None, :] < count[:, None]
        occ = ((lines > 0.2 * tile) & valid).sum(axis=1) / np.maximum(count, 1)
        req_share = np.where(b < 6, share_min, min(share_min, 0.72))
        ok = (share >= req_share) & (occ >= occ_min) & (count >= 3) & (total >= tile)
        score = np.where(ok, share * occ, 0)
        better = score > best_score
        best_score = np.where(better, score, best_score)
        best_b = np.where(better, b, best_b)
        best_p = np.where(better, p, best_p)
    return best_b, best_p, best_score > 0


def _tile_hits(ex: np.ndarray, ey: np.ndarray, tile: int, lo: int, hi: int,
               share_min: float = 0.80, occ_min: float = 0.45):
    """Tiles whose horizontal and vertical edges both sit on one block lattice: (y0, x0, b, gx, gy)."""
    h, w = ex.shape[0], ey.shape[1]
    stride = tile // 2
    ys = list(range(0, h - tile + 1, stride))
    xs = list(range(0, w - tile + 1, stride))
    if not ys or not xs:
        return []
    n = tile - 1
    csx = np.vstack([np.zeros((1, ex.shape[1]), np.int32), np.cumsum(ex, axis=0, dtype=np.int32)])
    csy = np.hstack([np.zeros((ey.shape[0], 1), np.int32), np.cumsum(ey, axis=1, dtype=np.int32)])
    cx = np.empty((len(ys), len(xs), n), np.float32)
    cy = np.empty((len(ys), len(xs), n), np.float32)
    for iy, y0 in enumerate(ys):
        rows = csx[y0 + tile] - csx[y0]
        for ix, x0 in enumerate(xs):
            cx[iy, ix] = rows[x0:x0 + n]
            cy[iy, ix] = csy[y0:y0 + n, x0 + tile] - csy[y0:y0 + n, x0]
    bx, px, okx = _lattice(cx.reshape(-1, n), tile, lo, hi, share_min, occ_min)
    by, py, oky = _lattice(cy.reshape(-1, n), tile, lo, hi, share_min, occ_min)
    hits = []
    for k in np.flatnonzero(okx & oky & (bx == by)):
        iy, ix = divmod(int(k), len(xs))
        y0, x0, b = ys[iy], xs[ix], int(bx[k])
        # An edge after pixel i means a block starts at i + 1.
        hits.append((y0, x0, b, (x0 + int(px[k]) + 1) % b, (y0 + int(py[k]) + 1) % b))
    return hits


def detect_mosaic(rgb: np.ndarray, min_block: int = 4, max_block: int = 32, tols: tuple[int, ...] = (3, 8),
                  min_blocks: int = 4):
    """Returns (regions, mask). `tols` are the colour steps that count as an edge: a strict one for lossless
    images and a lenient one for JPEG noise. The mask is the union of the mosaic blocks."""
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if min(h, w) < 32:
        return [], mask
    img = rgb.astype(np.int16)
    step_x = np.abs(np.diff(img, axis=1)).max(axis=2)
    step_y = np.abs(np.diff(img, axis=0)).max(axis=2)

    clusters: dict[tuple[int, int, int], set[tuple[int, int, int]]] = {}
    for tol in tols:
        ex, ey = step_x > tol, step_y > tol
        for tile in (32, 48, 96):
            hi = min(max_block, tile // 2)
            if hi < min_block:
                continue
            for y0, x0, b, gx, gy in _tile_hits(ex, ey, tile, min_block, hi, share_min=0.80, occ_min=0.45):
                clusters.setdefault((b, gx, gy), set()).add((y0, x0, tile))

    regions: list[Region] = []
    for (b, gx, gy), tiles in clusters.items():
        for region, piece, (sx, sy) in _refine(rgb, b, gx, gy, sorted(tiles), min_blocks):
            mask[sy:sy + piece.shape[0], sx:sx + piece.shape[1]][piece] = 255
            regions.append(region)
    return _merge_regions(regions), mask


def _refine(rgb: np.ndarray, b: int, gx: int, gy: int, tiles: list[tuple[int, int, int]], min_blocks: int):
    """Block-level pass for one lattice: flood out from the seed tiles over flat, varying blocks."""
    h, w = rgb.shape[:2]
    reach = lambda t: max(t * 3, 200)  # generous search window around confirmed seed tiles
    rx0 = max(min(x - reach(t) for _, x, t in tiles), 0)
    ry0 = max(min(y - reach(t) for y, _, t in tiles), 0)
    rx1 = min(max(x + t + reach(t) for _, x, t in tiles), w)
    ry1 = min(max(y + t + reach(t) for y, _, t in tiles), h)
    sx = rx0 + (gx - rx0) % b  # first block start at or after rx0 (blocks start where x % b == gx)
    sy = ry0 + (gy - ry0) % b
    nbx, nby = (rx1 - sx) // b, (ry1 - sy) // b
    if nbx < 2 or nby < 2:
        return []
    blocks = rgb[sy:sy + nby * b, sx:sx + nbx * b].astype(np.float32).reshape(nby, b, nbx, b, 3)
    mean = blocks.mean(axis=(1, 3))
    dev = np.abs(blocks - mean[:, None, :, None, :]).mean(axis=(1, 3, 4))

    seed = np.zeros((nby, nbx), bool)
    near = np.zeros((nby, nbx), bool)  # within half a tile (plus a block) of a seed tile
    for y0, x0, t in tiles:
        seed[max(-(-(y0 - sy) // b), 0):max((y0 + t - sy) // b, 0),
             max(-(-(x0 - sx) // b), 0):max((x0 + t - sx) // b, 0)] = True
        r = t // 3 + b // 2
        near[max((y0 - r - sy) // b, 0):max((y0 + t + r - sy) // b + 1, 0),
             max((x0 - r - sx) // b, 0):max((x0 + t + r - sx) // b + 1, 0)] = True
    if not seed.any():
        return []

    # Calibrated flat tolerance based on median dev of confirmed seed blocks
    flat_tol = float(np.clip(2.0 * np.median(dev[seed]) + 0.75, 0.75, 14.0))
    cand = dev <= flat_tol
    if flat_tol > 2.2:  # lossy source: the tolerance also admits smooth gradients, so do not let the flood run far
        cand &= near

    # A mosaic block differs from at least one neighbouring block; plain background does not.
    vary_tol = max(0.5, flat_tol)
    dx = np.abs(mean[:, 1:] - mean[:, :-1]).max(axis=2) > vary_tol
    dy = np.abs(mean[1:] - mean[:-1]).max(axis=2) > vary_tol
    varies = np.zeros_like(cand)
    varies[:, 1:] |= (dx & cand[:, 1:] & cand[:, :-1])
    varies[:, :-1] |= (dx & cand[:, 1:] & cand[:, :-1])
    varies[1:] |= (dy & cand[1:] & cand[:-1])
    varies[:-1] |= (dy & cand[1:] & cand[:-1])
    core = cv2.morphologyEx(varies.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    grid = (core.astype(bool) & cand).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=4)
    seed_comps = []
    for k in range(1, count):
        comp = labels == k
        if (comp & seed).any() and stats[k, cv2.CC_STAT_AREA] >= min_blocks and varies[comp].mean() >= 0.25:
            seed_comps.append(k)

    if not seed_comps:
        return []

    merged_mask = np.zeros((nby, nbx), bool)
    for sc in seed_comps:
        merged_mask |= (labels == sc)

    # For clean/lossless sources, expand to adjacent components on same lattice across highlights/splashes
    if flat_tol <= 4.0:
        for k in range(1, count):
            if k in seed_comps:
                continue
            comp = labels == k
            if stats[k, cv2.CC_STAT_AREA] < min_blocks or varies[comp].mean() < 0.25:
                continue
            dist = cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8))
            if (dist & merged_mask).any():
                merged_mask |= comp

    # Emit the component pieces
    y_idx, x_idx = np.where(merged_mask)
    if len(y_idx) == 0:
        return []
    bx0, by0 = int(x_idx.min()), int(y_idx.min())
    bx1, by1 = int(x_idx.max()) + 1, int(y_idx.max()) + 1
    bw, bh = bx1 - bx0, by1 - by0
    sub_comp = merged_mask[by0:by1, bx0:bx1]
    piece = np.kron(sub_comp, np.ones((b, b), bool))
    region = Region(sx + bx0 * b, sy + by0 * b, bw * b, bh * b, "mosaic", b)
    return [(region, piece, (sx + bx0 * b, sy + by0 * b))]


def _merge_regions(regions: list[Region]) -> list[Region]:
    """Drop regions fully inside another of the same kind (overlapping clusters of one mosaic)."""
    out: list[Region] = []
    for r in sorted(regions, key=lambda r: -r.w * r.h):
        if not any(o.kind == r.kind and o.x <= r.x and o.y <= r.y and o.x + o.w >= r.x + r.w and o.y + o.h >= r.y + r.h
                   for o in out):
            out.append(r)
    return sorted(out, key=lambda r: (r.y, r.x))


# ------------------------------------------------------------------ bars

def detect_bars(rgb: np.ndarray, dark: int = 16, min_thick: int = 10, min_len: int = 30,
                fill: float = 0.93, max_frac: float = 0.3, flat_range: int = 8):
    """Solid near-black rectangles (the classic censor bar). Thin panel lines and ragged black fills are rejected."""
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    black = (rgb.max(axis=2) <= dark).astype(np.uint8)
    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, np.ones((min_thick, min_thick), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(black, connectivity=8)
    regions: list[Region] = []
    for k in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[k])
        if max(bw, bh) < min_len or area / (bw * bh) < fill or bw * bh > max_frac * h * w:
            continue
        if bw >= 0.9 * w or bh >= 0.9 * h:  # a page border or panel frame, not a bar
            continue
        comp = labels == k
        if (rgb[comp].max(axis=1) <= flat_range).mean() < 0.9:  # mostly truly black: a dark gradient is not a bar
            continue
        mask[comp] = 255
        regions.append(Region(x, y, bw, bh, "bar"))
    return regions, mask


# ------------------------------------------------------------------ white censor bars and spots

def detect_white_censor(rgb: np.ndarray, min_area: int = 80, max_frac: float = 0.25):
    """Solid near-white censor stickers/dots or white censor bars on figures.
    Rejects normal page white borders and flat highlights by checking surrounding contrast and hue."""
    h, w = rgb.shape[:2]
    bright = (rgb.min(axis=2) >= 222).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    regions: list[Region] = []
    mask = np.zeros((h, w), np.uint8)
    for k in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[k])
        if area < min_area or bw * bh > max_frac * h * w:
            continue
        if bw >= 0.9 * w or bh >= 0.9 * h:
            continue
        comp = (labels == k).astype(np.uint8)
        dilated = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        ring = (dilated > 0) & (comp == 0)
        if ring.sum() < 20:
            continue
        ring_rgb = rgb[ring]
        contrast = float(rgb[comp == 1].mean()) - float(ring_rgb.mean())
        if contrast < 20:
            continue
        if (ring_rgb.min(axis=1) >= 215).mean() > 0.40:
            continue

        fill_ratio = area / (bw * bh)
        r_mean = ring_rgb[:, 0].mean()
        g_mean = ring_rgb[:, 1].mean()
        b_mean = ring_rgb[:, 2].mean()
        ring_redness = r_mean - max(g_mean, b_mean)

        is_white_bar = (fill_ratio >= 0.82) and (max(bw, bh) >= 30) and (min(bw, bh) >= 10)
        is_censor_spot = (fill_ratio >= 0.55) and (ring_redness >= 16) and (area >= 120) and (max(bw, bh) >= 20)

        if is_white_bar or is_censor_spot:
            pad = 6
            px, py = max(0, x - pad), max(0, y - pad)
            pw, ph = min(w - px, bw + 2 * pad), min(h - py, bh + 2 * pad)
            mask[comp == 1] = 255
            regions.append(Region(px, py, pw, ph, "bar" if is_white_bar else "spot"))
    return regions, mask


# ------------------------------------------------------------------ combining

def detect_all(rgb: np.ndarray, kinds: tuple[str, ...] = ("mosaic", "bars", "spots"), **mosaic_args):
    regions: list[Region] = []
    mask = np.zeros(rgb.shape[:2], np.uint8)
    if "mosaic" in kinds:
        r, m = detect_mosaic(rgb, **mosaic_args)
        regions += r
        mask |= m
    if "bars" in kinds or "black" in kinds:
        r, m = detect_bars(rgb)
        regions += r
        mask |= m
    if "bars" in kinds or "spots" in kinds or "white" in kinds:
        r, m = detect_white_censor(rgb)
        if "spots" not in kinds and "white" not in kinds:
            r = [reg for reg in r if reg.kind == "bar"]
        elif "bars" not in kinds and "white" not in kinds:
            r = [reg for reg in r if reg.kind == "spot"]
        regions += r
        mask |= m
    return _merge_regions(regions), mask


def manual_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]):
    """Mask and regions from (x, y, w, h) boxes, clipped to the image."""
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    regions = []
    for x, y, bw, bh in boxes:
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
            regions.append(Region(x0, y0, x1 - x0, y1 - y0, "manual"))
    return regions, mask


def grow_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Dilate so the inpaint covers the ragged edge blocks that did not fit the grid, and close small gaps."""
    if pixels <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    return cv2.dilate(mask, k)


def preview(rgb: np.ndarray, regions: list[Region], mask: np.ndarray) -> np.ndarray:
    """The image with the mask tinted and each region boxed and numbered."""
    out = rgb.copy()
    tint = np.array([255, 64, 64], np.float32)
    sel = mask > 0
    out[sel] = (out[sel] * 0.55 + tint * 0.45).astype(np.uint8)
    for i, r in enumerate(regions, 1):
        cv2.rectangle(out, (r.x, r.y), (r.x + r.w, r.y + r.h), (255, 255, 0), max(2, min(rgb.shape[:2]) // 400))
        cv2.putText(out, str(i), (r.x + 4, r.y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    return out
