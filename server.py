import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from notify import StreamAlerts, Telegram
from pipeline import DEVICE, Pipeline, TrackState

ROOT = Path(__file__).parent
app = FastAPI(title="Safety Vision")
pipe: Pipeline | None = None
tg = Telegram()
# single worker: MPS models are not safe to run concurrently and frames must stay ordered
pool = ThreadPoolExecutor(max_workers=1)


@app.on_event("startup")
def load() -> None:
    global pipe
    pipe = Pipeline()
    print(f"models loaded on {DEVICE}")
    print("telegram alerts:", "on" if tg.token else "off (set TELEGRAM_TOKEN in .env)")


def decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "ไฟล์ไม่ใช่รูปภาพ")
    return img


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    img = decode(await file.read())
    return await asyncio.get_running_loop().run_in_executor(pool, pipe.analyze, img, None)


@app.websocket("/ws/live")
async def live(ws: WebSocket):
    await ws.accept()
    state = TrackState()
    alerts = StreamAlerts(tg)
    flags = {"fire": True, "fall": True, "emotion": True}
    loop = asyncio.get_running_loop()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text"):  # module toggles from the UI
                flags.update({k: bool(v) for k, v in json.loads(msg["text"]).items() if k in flags})
                continue
            img = decode(msg["bytes"])
            res = await loop.run_in_executor(pool, lambda: pipe.analyze(img, state, **flags))
            await ws.send_json(res)
            alerts.check(img, res)
    except WebSocketDisconnect:
        pass


@app.get("/api/samples")
def samples():
    return sorted(p.name for p in (ROOT / "samples").glob("*.jpg"))


app.mount("/samples", StaticFiles(directory=ROOT / "samples"), name="samples")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
