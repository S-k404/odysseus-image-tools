"""Run and compare all three decensoring options on CG13_01.png_:
Option A: Pure Animagine XL Diffusion Inpainting (generative anatomical prompt)
Option B: Two-Stage Hybrid (LaMa base to clean blocks + Animagine XL detailer pass)
Option C: Er0manga Inpainting (FFC ResNet fine-tuned directly on adult manga)
"""
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
er0_dir = ROOT / "models" / "Er0mangaInpaint"
sys.path.insert(0, str(er0_dir))

from uncensor.pipeline import load_rgb
from uncensor.detect import detect_all, grow_mask
from uncensor.inpaint import Inpainter, window_for, _round8, NATIVE, POSITIVE_ANIME_MALE, NEGATIVE_ANIME

OUT_DIR = (Path.home() / "Pictures/odysseus-uncensor/TEST_COMPARISON")
OUT_DIR.mkdir(parents=True, exist_ok=True)

target_image = (Path.home() / "Documents/TEST/pictures/CG13_01.png_")
print(f"Loading target image: {target_image.name}")
rgb = load_rgb(target_image)
regions, raw_mask = detect_all(rgb)
print(f"Detected {len(regions)} regions.")

# Pad mask to cover block edges completely
pad = max([r.block or 8 for r in regions]) // 2 + 4
mask = grow_mask(raw_mask, pad)

# Save mask preview
cv2.imwrite(str(OUT_DIR / "CG13_01_mask.png"), mask)
print("Saved mask to CG13_01_mask.png")

# -------------------------------------------------------------------------
# OPTION C: Er0manga Inpainting (Fast FFC trained on adult manga)
# -------------------------------------------------------------------------
print("\n" + "="*50)
print("RUNNING OPTION C: Er0manga Inpainting (FFC ResNet on Adult Manga)")
print("="*50)
t0 = time.time()

from saicinpainting.training.modules import make_generator

with open(er0_dir / "pretrained/00-30-09/config.yaml") as f:
    config = OmegaConf.create(yaml.safe_load(f))

er0_gen = make_generator(config, **config.generator)
ckpt_path = er0_dir / "pretrained/00-30-09/models/best.ckpt"
data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
gen_sd = {k.replace("generator.", ""): v for k, v in data["state_dict"].items() if k.startswith("generator.")}
er0_gen.load_state_dict(gen_sd, strict=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
er0_gen.to(device)
er0_gen.eval()

# Run Er0manga inpainting
h, w = rgb.shape[:2]
pad_h = (8 - h % 8) % 8
pad_w = (8 - w % 8) % 8

img_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)

if pad_h > 0 or pad_w > 0:
    img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode="reflect")
    mask_t = F.pad(mask_t, (0, pad_w, 0, pad_h), mode="replicate")

masked_img = img_t * (1.0 - mask_t)
inp_4ch = torch.cat([masked_img, mask_t], dim=1).to(device)

with torch.no_grad():
    out_c_t = er0_gen(inp_4ch)

out_c_np = out_c_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
if pad_h > 0 or pad_w > 0:
    out_c_np = out_c_np[:h, :w]
out_c_img = np.clip(out_c_np * 255.0, 0, 255).astype(np.uint8)

# Feather blend
soft = mask.astype(np.float32) / 255.0
alpha = np.maximum(soft, cv2.GaussianBlur(soft, (0, 0), 3.0))[..., None]
res_c = (out_c_img.astype(np.float32) * alpha + rgb.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

out_c_path = OUT_DIR / "CG13_01_Option_C_Er0manga.png"
Image.fromarray(res_c).save(out_c_path)
t_c = time.time() - t0
print(f"Option C finished in {t_c:.2f}s -> Saved to {out_c_path.name}")

# Free er0_gen memory before running diffusion
del er0_gen, data, gen_sd
if device == "mps":
    torch.mps.empty_cache()

# -------------------------------------------------------------------------
# OPTION A: Animagine XL Diffusion Inpainting
# -------------------------------------------------------------------------
print("\n" + "="*50)
print("RUNNING OPTION A: Animagine XL Generative Diffusion Inpainting")
print("="*50)
t0 = time.time()

inpainter = Inpainter(model_dir=ROOT / "models" / "animagine-xl-inpainting")
res_a = inpainter.inpaint(
    rgb, mask,
    prompt="masterpiece, best quality, detailed anime penis, detailed glans, foreskin, veins, shaft, clean manga lineart, flawless anatomy",
    negative=NEGATIVE_ANIME,
    steps=30,
    guidance=7.5,
    seed=42,
    model=ROOT / "models" / "animagine-xl-inpainting",
    anatomy="male",
    style="anime"
)

out_a_path = OUT_DIR / "CG13_01_Option_A_AnimagineXL.png"
Image.fromarray(res_a).save(out_a_path)
t_a = time.time() - t0
print(f"Option A finished in {t_a:.2f}s -> Saved to {out_a_path.name}")

# -------------------------------------------------------------------------
# OPTION B: Two-Stage Hybrid (LaMa Base + Animagine XL Detailer)
# -------------------------------------------------------------------------
print("\n" + "="*50)
print("RUNNING OPTION B: Two-Stage Hybrid (LaMa Base + Detailer Pass)")
print("="*50)
t0 = time.time()

# Stage 1: Fast LaMa base
lama_model = torch.jit.load(str(ROOT / "models" / "anime-manga-big-lama.pt"), map_location=device)
lama_model.eval()

img_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)

if pad_h > 0 or pad_w > 0:
    img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode="reflect")
    mask_t = F.pad(mask_t, (0, pad_w, 0, pad_h), mode="replicate")

img_t = img_t * (1.0 - mask_t)
with torch.no_grad():
    lama_out_t = lama_model(img_t.to(device), mask_t.to(device))

lama_out_np = lama_out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
if pad_h > 0 or pad_w > 0:
    lama_out_np = lama_out_np[:h, :w]
lama_base = np.clip(lama_out_np * 255.0, 0, 255).astype(np.uint8)
lama_clean = (lama_base.astype(np.float32) * alpha + rgb.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

del lama_model
if device == "mps":
    torch.mps.empty_cache()

# Stage 2: Detailer pass with Animagine XL at low denoise (strength=0.60, steps=20)
res_b = inpainter.inpaint(
    lama_clean, mask,
    prompt="masterpiece, best quality, detailed anime penis, detailed glans, foreskin, veins, shaft, clean manga lineart, flawless anatomy",
    negative=NEGATIVE_ANIME,
    steps=20,
    strength=0.60,
    guidance=7.0,
    seed=42,
    model=ROOT / "models" / "animagine-xl-inpainting",
    anatomy="male",
    style="anime"
)

out_b_path = OUT_DIR / "CG13_01_Option_B_Hybrid.png"
Image.fromarray(res_b).save(out_b_path)
t_b = time.time() - t0
print(f"Option B finished in {t_b:.2f}s -> Saved to {out_b_path.name}")

print("\n" + "="*50)
print("ALL THREE OPTIONS COMPLETED!")
print(f"Option A (Animagine XL):    {t_a:.1f}s -> {out_a_path}")
print(f"Option B (Two-Stage Hybrid):{t_b:.1f}s -> {out_b_path}")
print(f"Option C (Er0manga FFC):    {t_c:.1f}s -> {out_c_path}")
print("="*50)
