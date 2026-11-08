"""Run all three options on CG12_03.png_ (female genitalia mosaic)
"""
import sys, time, cv2, torch, yaml
import numpy as np
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
er0_dir = ROOT / "models" / "Er0mangaInpaint"
sys.path.insert(0, str(er0_dir))

from uncensor.pipeline import load_rgb
from uncensor.detect import detect_all, grow_mask
from uncensor.inpaint import Inpainter, NEGATIVE_ANIME

OUT_DIR = (Path.home() / "Pictures/odysseus-uncensor/TEST_COMPARISON")

target = (Path.home() / "Documents/TEST/pictures/CG12_03.png_")
rgb = load_rgb(target)
regions, raw_mask = detect_all(rgb)
pad = max([r.block or 8 for r in regions]) // 2 + 4
mask = grow_mask(raw_mask, pad)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# ----------------- Option C: Er0manga -----------------
from saicinpainting.training.modules import make_generator
with open(er0_dir / "pretrained/00-30-09/config.yaml") as f:
    config = OmegaConf.create(yaml.safe_load(f))
er0_gen = make_generator(config, **config.generator)
data = torch.load(er0_dir / "pretrained/00-30-09/models/best.ckpt", map_location="cpu", weights_only=False)
gen_sd = {k.replace("generator.", ""): v for k, v in data["state_dict"].items() if k.startswith("generator.")}
er0_gen.load_state_dict(gen_sd, strict=True)
er0_gen.to(device)
er0_gen.eval()

h, w = rgb.shape[:2]
pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
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

soft = mask.astype(np.float32) / 255.0
alpha = np.maximum(soft, cv2.GaussianBlur(soft, (0, 0), 3.0))[..., None]
res_c = (out_c_img.astype(np.float32) * alpha + rgb.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
Image.fromarray(res_c).save(OUT_DIR / "CG12_03_Option_C_Er0manga.png")

del er0_gen, data, gen_sd
if device == "mps":
    torch.mps.empty_cache()

# ----------------- Option A: Animagine XL -----------------
inpainter = Inpainter(model_dir=ROOT / "models" / "animagine-xl-inpainting")
res_a = inpainter.inpaint(
    rgb, mask,
    prompt="masterpiece, best quality, uncensored, detailed anime pussy, clitoris, labia, vaginal opening, wet, clean lineart, flawless anatomy",
    negative=NEGATIVE_ANIME,
    steps=30,
    guidance=7.5,
    seed=42,
    model=ROOT / "models" / "animagine-xl-inpainting",
    anatomy="female",
    style="anime"
)
Image.fromarray(res_a).save(OUT_DIR / "CG12_03_Option_A_AnimagineXL.png")

# ----------------- Option B: Two-Stage Hybrid -----------------
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

res_b = inpainter.inpaint(
    lama_clean, mask,
    prompt="masterpiece, best quality, uncensored, detailed anime pussy, clitoris, labia, vaginal opening, wet, clean lineart, flawless anatomy",
    negative=NEGATIVE_ANIME,
    steps=20,
    strength=0.60,
    guidance=7.0,
    seed=42,
    model=ROOT / "models" / "animagine-xl-inpainting",
    anatomy="female",
    style="anime"
)
Image.fromarray(res_b).save(OUT_DIR / "CG12_03_Option_B_Hybrid.png")

# Save crops
c_orig = Image.fromarray(rgb).crop((380, 380, 520, 540))
c_a = Image.open(OUT_DIR / "CG12_03_Option_A_AnimagineXL.png").crop((380, 380, 520, 540))
c_b = Image.open(OUT_DIR / "CG12_03_Option_B_Hybrid.png").crop((380, 380, 520, 540))
c_c = Image.open(OUT_DIR / "CG12_03_Option_C_Er0manga.png").crop((380, 380, 520, 540))

w_c, h_c = c_orig.size
side = Image.new("RGB", (w_c * 4, h_c))
side.paste(c_orig, (0, 0))
side.paste(c_a, (w_c, 0))
side.paste(c_b, (w_c * 2, 0))
side.paste(c_c, (w_c * 3, 0))
side.save(OUT_DIR / "CG12_03_side_by_side.png")
print("CG12_03 tests completed and saved to TEST_COMPARISON!")
