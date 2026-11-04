"""Drive uncensor_server.py over stdio like Odysseus does, on a harmless mosaiced picture.

Run: .venv/bin/python tests/test_mcp.py <mosaic_image.png>   (takes about a minute; uses the GPU)
"""
import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent


async def main(image: str) -> int:
    out_dir = tempfile.mkdtemp(prefix="uncensor-test-")
    pages = Path(out_dir) / "pages"
    pages.mkdir()
    for n in (2, 10):  # natural sort: page2 before page10
        shutil.copy(image, pages / f"page{n}.png")
    params = StdioServerParameters(command=str(ROOT / ".venv/bin/python"), args=[str(ROOT / "uncensor_server.py")],
                                   env={"UNCENSOR_OUT_DIR": out_dir, "PATH": "/usr/bin:/bin"})
    failures = []

    def check(name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name + "  " + detail)
        if not ok:
            failures.append(name)

    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            names = [t.name for t in (await s.list_tools()).tools]
            check("tools listed", len(names) >= 5, ", ".join(names))
            call = lambda n, a: s.call_tool(n, a)
            text = lambda res: res.content[0].text

            t = text(await call("find_censored_regions", {"image_path": image}))
            check("find", "mosaic" in t and "preview:" in t, t.replace("\n", " | "))
            t = text(await call("find_censored_regions", {"image_path": "/nonexistent.png"}))
            check("bad path is an error text, not a crash", t.startswith("Error:"), t)
            t = text(await call("uncensor_image", {"image_path": image, "prompt": "a white lighthouse tower", "seed": 3, "steps": 20}))
            saved = [l for l in t.splitlines() if l.startswith("saved: ")]
            check("uncensor_image", bool(saved) and Path(saved[0][7:]).is_file(), t.replace("\n", " | "))
            t = text(await call("uncensor_folder", {"folder": str(pages), "steps": 15}))
            job = t.split()[1]
            check("folder job starts", t.startswith("job "), t)
            for _ in range(120):
                st = text(await call("uncensor_job_status", {"job_id": job}))
                if st.startswith("finished"):
                    break
                await asyncio.sleep(3)
            check("folder job finishes", st.startswith("finished: 2/2"), st.replace("\n", " | "))
            check("page order", st.index("page2") < st.index("page10"), "")
            t = text(await call("unload_uncensor_model", {}))
            check("unload", t == "unloaded", t)
    shutil.rmtree(out_dir, ignore_errors=True)
    print("\nFAILED: " + ", ".join(failures) if failures else "\nall passed")
    return 1 if failures else 0


import numpy as np
from PIL import Image


def _make_mosaic(img: np.ndarray, box, block: int = 12) -> np.ndarray:
    x0, y0, x1, y1 = box
    out = img.copy()
    for y in range(y0, y1, block):
        for x in range(x0, x1, block):
            ye, xe = min(y + block, y1), min(x + block, x1)
            out[y:ye, x:xe] = img[y:ye, x:xe].reshape(-1, 3).mean(axis=0).astype(np.uint8)
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        base = np.array(Image.open(ROOT / "tests/data/lighthouse512.png").convert("RGB"))
        mosaiced = _make_mosaic(base, (100, 90, 300, 210), 12)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(mosaiced).save(tmp.name)
        img_path = tmp.name
    sys.exit(asyncio.run(main(img_path)))
