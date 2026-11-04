"""
uncensor_server.py

MCP server for Odysseus: find mosaic / black-bar censoring in an image and fill it in with a local
inpainting model (SDXL on the Mac's GPU). Small tool schemas and compact outputs so local models
can drive it. Originals are never modified; results go to ~/Pictures/odysseus-uncensor.
"""

import asyncio
import sys
import threading
import time
import uuid
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uncensor import pipeline as pl  # noqa: E402
from uncensor.inpaint import Inpainter  # noqa: E402

server = Server("uncensor")
inpainter = Inpainter(idle_seconds=300)
JOBS: dict[str, dict] = {}
_idle_task: asyncio.Task | None = None

PROMPT_HELP = ("What should be in the covered area, in plain words (e.g. 'a white lighthouse tower'). "
               "The surroundings are used as well; leave empty to let the model continue the picture.")

TOOLS = [
    Tool(
        name="find_censored_regions",
        description="Look for mosaic (pixelation) and black censor bars in a manga panel or image. Changes nothing; writes a preview "
                    "with the regions outlined and numbered, and lists them with x, y, width, height.",
        inputSchema={"type": "object", "properties": {
            "image_path": {"type": "string", "description": "Full path of the image (png, jpg, webp, png_)"},
            "kinds": {"type": "string", "description": "mosaic, bars or 'mosaic,bars' (default both)"},
        }, "required": ["image_path"]},
    ),
    Tool(
        name="uncensor_image",
        description="Decensor / uncensor an image or manga panel by removing mosaic pixelation and black censor bars using AI inpainting (Animagine XL / SDXL). "
                    "It invents plausible detail, it cannot recover what was hidden. Finds the regions itself unless "
                    "'regions' or 'mask_path' is given. Saves a new file; the original is untouched. Takes 15-40 s.",
        inputSchema={"type": "object", "properties": {
            "image_path": {"type": "string", "description": "Full path of the image"},
            "prompt": {"type": "string", "description": PROMPT_HELP},
            "regions": {"type": "string", "description": "Optional: your own areas as 'x,y,w,h;x,y,w,h' (skips detection)"},
            "mask_path": {"type": "string", "description": "Optional: mask image, white = fill in (skips detection)"},
            "kinds": {"type": "string", "description": "What to detect: mosaic, bars or 'mosaic,bars' (default both)"},
            "model": {"type": "string", "description": "Model to use: 'anime' (default, specialized for manga/anime) or 'general' (SDXL photorealistic)"},
            "style": {"type": "string", "description": "Style prompt preset: 'auto' (default), 'anime' or 'general'"},
            "anatomy": {"type": "string", "description": "Anatomy prompt preset: 'auto' (default, auto-detects male vs female), 'female', 'male', or 'none'"},
            "steps": {"type": "integer", "description": "Quality vs speed, 15-50 (default 30)"},
            "seed": {"type": "integer", "description": "Repeat a result by reusing its seed (default random)"},
        }, "required": ["image_path"]},
    ),
    Tool(
        name="decensor_image",
        description="Alias for uncensor_image: Decensor / uncensor a manga panel or image by removing mosaic or black bars using AI inpainting.",
        inputSchema={"type": "object", "properties": {
            "image_path": {"type": "string", "description": "Full path of the image"},
            "prompt": {"type": "string", "description": PROMPT_HELP},
            "regions": {"type": "string", "description": "Optional: your own areas as 'x,y,w,h;x,y,w,h' (skips detection)"},
            "mask_path": {"type": "string", "description": "Optional: mask image, white = fill in (skips detection)"},
            "kinds": {"type": "string", "description": "What to detect: mosaic, bars or 'mosaic,bars' (default both)"},
            "model": {"type": "string", "description": "Model to use: 'anime' (default, specialized for manga/anime) or 'general' (SDXL photorealistic)"},
            "style": {"type": "string", "description": "Style prompt preset: 'auto' (default), 'anime' or 'general'"},
            "anatomy": {"type": "string", "description": "Anatomy prompt preset: 'auto' (default, auto-detects male vs female), 'female', 'male', or 'none'"},
            "steps": {"type": "integer", "description": "Quality vs speed, 15-50 (default 30)"},
            "seed": {"type": "integer", "description": "Repeat a result by reusing its seed (default random)"},
        }, "required": ["image_path"]},
    ),
    Tool(
        name="uncensor_folder",
        description="Decensor every manga panel / image in a folder (chapter, directory, or game pictures folder). Automatically "
                    "detects images with censorship, inpaints them, and saves to ~/Pictures/odysseus-uncensor. "
                    "Runs in the background and returns a job id; check progress with uncensor_job_status.",
        inputSchema={"type": "object", "properties": {
            "folder": {"type": "string", "description": "Full path of the folder of images / manga panels"},
            "prompt": {"type": "string", "description": PROMPT_HELP},
            "model": {"type": "string", "description": "Model to use: 'anime' (default, specialized for manga/anime) or 'general' (SDXL photorealistic)"},
            "style": {"type": "string", "description": "Style prompt preset: 'auto' (default), 'anime' or 'general'"},
            "anatomy": {"type": "string", "description": "Anatomy prompt preset: 'auto' (default), 'female', 'male', or 'none'"},
            "steps": {"type": "integer", "description": "Quality vs speed, 15-50 (default 30)"},
        }, "required": ["folder"]},
    ),
    Tool(
        name="decensor_folder",
        description="Alias for uncensor_folder: Decensor every manga panel / image in a folder.",
        inputSchema={"type": "object", "properties": {
            "folder": {"type": "string", "description": "Full path of the folder of images / manga panels"},
            "prompt": {"type": "string", "description": PROMPT_HELP},
            "model": {"type": "string", "description": "Model to use: 'anime' (default, specialized for manga/anime) or 'general' (SDXL photorealistic)"},
            "style": {"type": "string", "description": "Style prompt preset: 'auto' (default), 'anime' or 'general'"},
            "anatomy": {"type": "string", "description": "Anatomy prompt preset: 'auto' (default), 'female', 'male', or 'none'"},
            "steps": {"type": "integer", "description": "Quality vs speed, 15-50 (default 30)"},
        }, "required": ["folder"]},
    ),
    Tool(
        name="uncensor_job_status",
        description="Progress of an uncensor_folder / decensor_folder job: pages done, outputs, errors.",
        inputSchema={"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
    ),
    Tool(
        name="unload_uncensor_model",
        description="Free the memory the inpainting model uses (about 8 GB). It also unloads itself after 5 idle minutes.",
        inputSchema={"type": "object", "properties": {}},
    ),
]


def _kinds(arguments: dict) -> tuple[str, ...]:
    parts = tuple(k.strip() for k in str(arguments.get("kinds") or "mosaic,bars").split(",") if k.strip() in ("mosaic", "bars"))
    return parts or ("mosaic", "bars")


def _steps(arguments: dict) -> int:
    return max(10, min(int(arguments.get("steps") or 30), 60))


def _run_folder(job_id: str, folder: Path, prompt: str, steps: int, model: str = "anime", style: str = "auto", anatomy: str = "auto") -> None:
    job = JOBS[job_id]
    out_dir = pl.OUT_DIR / folder.name
    for i, path in enumerate(job["pages"], 1):
        job["current"] = path.name
        try:
            if path.suffix.lower() in pl.VIDEO_EXTS:
                res = pl.uncensor_video(path, inpainter, out_dir=out_dir, prompt=prompt, steps=steps, model=model, style=style, anatomy=anatomy)
                job["results"].append(f"{path.name}: {res['output']} ({res.get('censored_frames', 0)} frames, {res.get('seconds', 0)}s)")
                job["uncensored_count"] = job.get("uncensored_count", 0) + 1
            else:
                res = pl.uncensor(path, inpainter, out_dir=out_dir, prompt=prompt, steps=steps, model=model, style=style, anatomy=anatomy)
                if res.get("output"):
                    job["results"].append(f"{path.name}: {res['output']} ({len(res.get('regions', []))} regions, {res.get('seconds', 0)}s)")
                    job["uncensored_count"] = job.get("uncensored_count", 0) + 1
                else:
                    job["skipped_count"] = job.get("skipped_count", 0) + 1
        except Exception as exc:  # keep going: one bad page should not stop the chapter
            job["results"].append(f"{path.name}: ERROR {exc}")
        job["done"] = i
    job["state"] = "finished"
    job["current"] = ""
    job["finished_at"] = time.time()


def _handle(name: str, arguments: dict) -> str:
    if name == "find_censored_regions":
        path = pl.check_image_path(arguments.get("image_path", ""))
        rgb, regions, mask = pl.find_regions(path, _kinds(arguments))
        if not regions:
            return "No mosaic or black bars found. If something is covered, give the area yourself as regions='x,y,w,h'."
        preview = pl.write_preview(path, rgb, regions, mask)
        return "\n".join(pl.describe(regions) + [f"preview: {preview}"])

    if name in ("uncensor_image", "decensor_image"):
        path = pl.check_image_path(arguments.get("image_path", ""))
        if path.suffix.lower() in pl.VIDEO_EXTS:
            res = pl.uncensor_video(
                path, inpainter, prompt=str(arguments.get("prompt") or ""),
                steps=_steps(arguments),
                model=str(arguments.get("model") or "anime"),
                style=str(arguments.get("style") or "auto"),
                anatomy=str(arguments.get("anatomy") or "auto"),
            )
            return f"Decensored video saved: {res['output']} ({res.get('censored_frames', 0)} censored frames, {res.get('seconds', 0)}s)"

        mask = arguments.get("mask_path")
        res = pl.uncensor(
            path, inpainter, kinds=_kinds(arguments),
            boxes=pl.parse_boxes(arguments["regions"]) if arguments.get("regions") else None,
            mask_path=pl.check_image_path(mask) if mask else None,
            prompt=str(arguments.get("prompt") or ""), steps=_steps(arguments),
            seed=int(arguments["seed"]) if arguments.get("seed") is not None else None,
            model=str(arguments.get("model") or "anime"),
            style=str(arguments.get("style") or "auto"),
            anatomy=str(arguments.get("anatomy") or "auto"),
        )
        if not res.get("output"):
            return res["note"] + ("\n" + "\n".join(res["regions"]) if res.get("regions") else "")
        return "\n".join(["Filled in " + str(len(res["regions"])) + " region(s):"] + res["regions"]
                         + [f"saved: {res['output']}", f"time: {res['seconds']} s"])

    if name in ("uncensor_folder", "decensor_folder"):
        folder = Path(str(arguments.get("folder", ""))).expanduser()
        if not folder.is_dir():
            raise ValueError(f"no such folder: {folder}")
        pages = pl.list_images(folder)
        if not pages:
            raise ValueError("no images in that folder")
        job_id = uuid.uuid4().hex[:8]
        JOBS[job_id] = {"state": "running", "pages": pages, "done": 0, "current": "", "results": [], "started_at": time.time(), "uncensored_count": 0, "skipped_count": 0}
        threading.Thread(target=_run_folder, args=(job_id, folder, str(arguments.get("prompt") or ""), _steps(arguments),
                                                  str(arguments.get("model") or "anime"), str(arguments.get("style") or "auto"),
                                                  str(arguments.get("anatomy") or "auto")),
                         daemon=True).start()
        return f"job {job_id} started: {len(pages)} pages, results go to {pl.OUT_DIR / folder.name}"

    if name == "uncensor_job_status":
        job = JOBS.get(str(arguments.get("job_id", "")))
        if not job:
            raise ValueError("unknown job id")
        uncensored = job.get("uncensored_count", 0)
        head = f"{job['state']}: {job['done']}/{len(job['pages'])} pages ({uncensored} uncensored)" + (f", now {job['current']}" if job["current"] else "")
        return "\n".join([head] + job["results"][-15:])

    if name == "unload_uncensor_model":
        return "unloaded" if inpainter.unload() else "was not loaded"

    raise ValueError(f"Unknown tool: {name}")


async def _free_when_idle() -> None:
    while True:
        await asyncio.sleep(30)
        await asyncio.to_thread(inpainter.unload_if_idle)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    global _idle_task
    if _idle_task is None:
        _idle_task = asyncio.create_task(_free_when_idle())
    try:
        text = await asyncio.to_thread(_handle, name, arguments or {})
    except Exception as exc:
        text = f"Error: {exc}"
    return [TextContent(type="text", text=text)]


async def run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(run())
