#!/usr/bin/env python3
"""Command line front end for the uncensor tool.

  uncensor_cli.py detect PAGE.png                      # find regions, write a preview, change nothing
  uncensor_cli.py run PAGE.png [--prompt "..."]        # detect + fill in
  uncensor_cli.py run FOLDER/ [--prompt "..."]         # every image in the folder, in page order
  uncensor_cli.py run PAGE.png --box 120,340,200,160   # your own rectangle (x,y,w,h), repeatable

Outputs go to ~/Pictures/odysseus-uncensor (or $UNCENSOR_OUT_DIR); originals are never modified.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uncensor.inpaint import Inpainter  # noqa: E402
from uncensor import pipeline as pl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("path")
    r = sub.add_parser("run")
    r.add_argument("path")
    r.add_argument("--prompt", default="", help="what should be there (the surroundings are used too)")
    r.add_argument("--negative", default="")
    r.add_argument("--box", action="append", default=[], help="x,y,w,h; skips detection")
    r.add_argument("--mask", help="mask image, white = fill in; skips detection")
    r.add_argument("--steps", type=int, default=30)
    r.add_argument("--guidance", type=float, default=7.5)
    r.add_argument("--strength", type=float, default=0.99)
    r.add_argument("--grow", type=int, help="pixels to widen the mask by (default: from the block size)")
    r.add_argument("--seed", type=int)
    r.add_argument("--model", default="er0manga", help="model name or path ('er0manga' [default], 'hybrid', 'animagine', 'lama')")
    r.add_argument("--out-dir", default=None, help="custom output directory")
    r.add_argument("--style", default="auto", help="prompt style preset ('auto', 'anime', 'general')")
    r.add_argument("--anatomy", default="auto", choices=["auto", "female", "male", "none"],
                   help="anatomical prompt preset for anime decensoring ('auto', 'female', 'male', 'none')")
    a = ap.parse_args()

    target = Path(a.path).expanduser()
    if a.cmd == "detect":
        path = pl.check_image_path(a.path)
        rgb, regions, mask = pl.find_regions(path)
        print("\n".join(pl.describe(regions)) or "no mosaic or bars found")
        if regions:
            print("preview:", pl.write_preview(path, rgb, regions, mask))
        return 0

    paths = pl.list_images(target) if target.is_dir() else [pl.check_image_path(a.path)]
    out_dir = Path(a.out_dir).expanduser() if a.out_dir else (pl.OUT_DIR / target.name if target.is_dir() else pl.OUT_DIR)
    inpainter = Inpainter(model_dir=a.model)
    boxes = [pl.parse_boxes(b)[0] for b in a.box]
    uncensored_count = 0
    try:
        for i, path in enumerate(paths, 1):
            if path.suffix.lower() in pl.VIDEO_EXTS:
                print(f"[{i}/{len(paths)}] Processing video: {path.name} ...", flush=True)
                res = pl.uncensor_video(path, inpainter, out_dir=out_dir, prompt=a.prompt, negative=a.negative,
                                        steps=a.steps, guidance=a.guidance, strength=a.strength,
                                        model=a.model, style=a.style, anatomy=a.anatomy,
                                        progress=lambda m: print("   ", m, flush=True))
                uncensored_count += 1
                print(f"[{i}/{len(paths)}] {path.name} -> {res['output']} ({res.get('censored_frames', 0)} censored frames, {res.get('seconds', 0)}s)", flush=True)
                continue

            res = pl.uncensor(path, inpainter, out_dir=out_dir, boxes=boxes or None, mask_path=Path(a.mask).expanduser() if a.mask else None,
                              prompt=a.prompt, negative=a.negative, steps=a.steps, guidance=a.guidance,
                              strength=a.strength, grow=a.grow, seed=a.seed, model=a.model, style=a.style,
                              anatomy=a.anatomy, progress=lambda m: print("   ", m, flush=True))
            if res.get("output"):
                uncensored_count += 1
                print(f"[{i}/{len(paths)}] {path.name} -> {res['output']} ({len(res.get('regions', []))} region(s), {res.get('seconds', 0)}s)", flush=True)
            else:
                if len(paths) <= 20:
                    print(f"[{i}/{len(paths)}] {path.name}: {res.get('note')}", flush=True)
                elif i % 50 == 0 or i == len(paths):
                    print(f"[{i}/{len(paths)}] scanned... ({uncensored_count} uncensored so far)", flush=True)
        print(f"\nDone! Uncensored {uncensored_count} images. Results saved in {out_dir}")
    finally:
        inpainter.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
