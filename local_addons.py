"""Optional add-on loaded by server.py: open-palm photo booth for the Live page.

The page script (local_static/capture.js) detects an open hand, counts down and posts the annotated
frame here. The photo is saved to captures/ and uploaded to tmpfiles.org (deleted after 60 minutes) so a QR
code can point phones outside the local network at it.
"""
import asyncio
import secrets
import time
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

ROOT = Path(__file__).parent
CAPTURES = ROOT / "captures"
SCRIPT = ROOT / "local_static" / "capture.js"
UPLOAD = "https://tmpfiles.org/api/v1/upload"  # files are deleted after 60 minutes; litterbox/0x0.st refused uploads (Nov 2026)
MAX_BYTES = 10 * 2**20


def upload(data: bytes, name: str) -> str:
    """Returns the tmpfiles page URL (it has a download button; the /dl/ link redirects to it anyway)."""
    r = requests.post(UPLOAD, files={"file": (name, data, "image/jpeg")}, timeout=30)
    try:
        url = r.json()["data"]["url"]
    except (ValueError, KeyError, TypeError):
        raise RuntimeError(f"{r.status_code} {r.text[:200]}")
    if not url.startswith("https://"):
        raise RuntimeError(f"unexpected url {url[:200]}")
    return url


def install(app: FastAPI) -> None:
    CAPTURES.mkdir(exist_ok=True)

    @app.get("/", include_in_schema=False)
    def index():
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        tag = f'<script type="module" src="/local/capture.js?v={int(SCRIPT.stat().st_mtime)}"></script>\n</body>'
        return HTMLResponse(html.replace("</body>", tag, 1), headers={"Cache-Control": "no-cache"})

    @app.get("/local/capture.js", include_in_schema=False)
    def script():
        return FileResponse(SCRIPT, media_type="text/javascript")

    @app.post("/api/capture")
    async def capture(request: Request):
        data = await request.body()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "รูปใหญ่เกินไป")
        if not data.startswith(b"\xff\xd8\xff"):
            raise HTTPException(400, "ต้องเป็นไฟล์ JPEG")
        name = time.strftime("%Y%m%d-%H%M%S") + f"-{secrets.token_hex(3)}.jpg"
        (CAPTURES / name).write_bytes(data)
        try:
            url = await asyncio.to_thread(upload, data, name)
        except (requests.RequestException, RuntimeError) as e:
            print("capture upload:", e)
            return {"file": name, "url": None, "error": "อัพโหลดไม่สำเร็จ ตรวจอินเทอร์เน็ต"}
        return {"file": name, "url": url}
