# Odysseus image tools

Two things for the local Odysseus setup (`~/Documents/Docker/Odysseus`), kept out of Odysseus's own folders and its `.venv`:

1. **Image generation** – already wired, nothing in this folder (see below).
2. **Uncensor** – an MCP server (`uncensor`) that finds mosaic / black-bar censoring in an image and fills it in with a local
   inpainting model. Also a command line tool.

## Uncensor

What it is, honestly: detection is classical image analysis (no ML). The fill-in is **SDXL inpainting** (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`,
OpenRAIL++ licence) running on the Mac's GPU. It **invents** plausible content that fits the surroundings; it cannot recover what the
mosaic hid. The description you give (`prompt`) decides what it draws. There is no safety filter in the pipeline. Use it on your own
copies of material with adult characters.

### From Odysseus

Tools (the local model calls them): `find_censored_regions`, `uncensor_image`, `uncensor_folder`, `uncensor_job_status`, `unload_uncensor_model`.
Registered in Odysseus's DB as server id `uncensor` (restart Odysseus.app after changing the server). Example task for the model:

> Uncensor `/path/page_012.png`. The covered area should show <plain description>.

Results are new files in `~/Pictures/odysseus-uncensor/` (`<name>_uncensored.png`, and `<name>_regions.png` from detection). **Originals are never touched**,
and nothing is overwritten (`_2`, `_3` … is added). `$UNCENSOR_OUT_DIR` changes the folder.

### From the terminal

```sh
cd ~/Documents/Docker/odysseus-image-tools
.venv/bin/python uncensor_cli.py detect PAGE.png                       # what would be filled in; writes a preview, changes nothing
.venv/bin/python uncensor_cli.py run PAGE.png --prompt "what belongs there"
.venv/bin/python uncensor_cli.py run CHAPTER_FOLDER/ --prompt "..."    # every page, in page order
.venv/bin/python uncensor_cli.py run PAGE.png --box 120,340,200,160    # your own rectangle x,y,w,h (repeatable); skips detection
.venv/bin/python uncensor_cli.py run PAGE.png --mask mask.png          # white = fill in
```

Options: `--steps` (30; fewer = faster), `--seed` (repeat a result), `--grow` (pixels to widen the mask; default from the block size),
`--guidance`, `--strength`, `--negative`.

### What it finds, and what it does not

- **Mosaic**: found by its block grid (block size 4–32 px, any offset), also in JPEGs. Tested on synthetic mosaics with known boxes: IoU 0.82–1.00.
  Weak spots: a mosaic over a very smooth gradient (blocks barely differ) can be found only in part, and JPEG over smooth gradients can
  over-cover by tens of pixels. Check the `_regions.png` preview, or pass your own `regions`.
- **Black bars**: solid near-black rectangles. **Not** white bars, coloured bars, blur, or pixelation drawn over curved shapes: give those as `regions` or a `mask`.
- Page-sized black fills and thin panel lines are ignored. No false hits on the clean test images.

### Resources

- Weights ≈ 6.6 GB in `models/sdxl-inpainting` (fp16 UNet and text encoders, fp32 VAE; safetensors only). Loaded when first used, dropped after 5 idle minutes
  or with `unload_uncensor_model`. About 8 GB of RAM while loaded: on this 48 GB Mac with the Qwen 35B model resident, memory pressure
  reached "warning". Unload the chat model first (menu bar → Unload) for big batches.
- Speed on the M5 Max: ~12 s for one region at 1024 px after the model is loaded, ~20 s including the first load.
- `UNCENSOR_MODEL_DIR` points to another diffusers **SDXL inpainting** model if you have a better-suited one (anime fine-tunes exist); it must be safetensors.

### Files

| File | |
|---|---|
| `uncensor_server.py` | the MCP server |
| `uncensor_cli.py` | command line front end |
| `uncensor/detect.py` | mosaic and bar detection, mask and preview |
| `uncensor/inpaint.py` | the SDXL inpainting step (crops each region with context, blends back with a soft edge) |
| `uncensor/pipeline.py` | detect + fill + write; shared by both front ends |
| `setup.sh`, `download_models.py` | create `.venv` (pins `mcp<2`, matching Odysseus's server style) and fetch the weights |
| `tests/test_detect.py` | detector vs known ground truth on harmless synthetic data (11 checks) |
| `tests/test_mcp.py <mosaic.png>` | drives the server over stdio like Odysseus (find, fill, folder job, unload) |

Rebuild from scratch: `./setup.sh && .venv/bin/python download_models.py`.

## Image generation (already set up)

Odysseus's built-in `image_gen` tool needed an image endpoint and had none (it expected OpenAI's gpt-image / dall-e). Docker Model Runner already had
`docker.io/ai/stable-diffusion` pulled and serves `/v1/images/generations`, so:

- `model_endpoints` row `dmr_image` ("Local Image Model (Docker Model Runner)", `http://127.0.0.1:12434/v1`, type `image`) was added to `data/app.db`.
- `data/settings.json`: `image_model` = `docker.io/ai/stable-diffusion:latest` (was `stable-diffusion-3.5-medium`, which had no weights).
- Verified through Odysseus's own `image_gen` MCP server: 1024×1024 in ~37 s, saved to the gallery. Needs Docker Model Runner running (it is, on port 12434).
- Backups: `data/app.db.bak_imagegen_*`, `data/settings.json.bak_imagegen_*`, `data/app.db.bak_uncensor_*`. Undo = delete the `dmr_image` / `uncensor` rows and restore `image_model`.
- Not yet driven from the chat window by the Qwen model (loading it alongside the image model is heavy on memory); the tool path below it is tested.
