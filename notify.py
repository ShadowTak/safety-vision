"""Telegram photo alerts for the live stream: a fall scored >= 80% or fire.

Config (.env, loaded by run.sh): TELEGRAM_TOKEN, optional TELEGRAM_CHAT_ID. Without a chat ID the
chat that last sent the bot a message is used and saved to .telegram_chat_id.
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests

API = "https://api.telegram.org/bot{token}/{method}"
CHAT_FILE = Path(__file__).parent / ".telegram_chat_id"
FALL_NOTIFY = 0.8   # fall_score needed to notify (the on-screen alarm is lower, 0.5)
CONFIRM_FRAMES = 2  # consecutive live detections before notifying
COOLDOWN_S = 60     # at most one message per kind per minute


class Telegram:
    def __init__(self) -> None:
        self.token = os.environ.get("TELEGRAM_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID") or (
            CHAT_FILE.read_text().strip() if CHAT_FILE.exists() else None)
        self._pool = ThreadPoolExecutor(max_workers=1)  # never block inference on the network
        self._last: dict[str, float] = {}  # kind -> last send time, shared by all streams
        self._lock = threading.Lock()

    def send(self, kind: str, caption: str, img: np.ndarray, res: dict) -> None:
        """Queue a photo alert unless this kind was sent within COOLDOWN_S."""
        now = time.time()
        with self._lock:
            if not self.token or now - self._last.get(kind, 0) < COOLDOWN_S:
                return
            self._last[kind] = now
        self._pool.submit(self._send, caption, img, res)

    def _chat(self) -> str | None:
        if self.chat_id:
            return self.chat_id
        r = requests.get(API.format(token=self.token, method="getUpdates"), timeout=10).json()
        chats = [u["message"]["chat"]["id"] for u in r.get("result", []) if "message" in u]
        if not chats:
            return None
        self.chat_id = str(chats[-1])
        CHAT_FILE.write_text(self.chat_id)
        return self.chat_id

    def _send(self, caption: str, img: np.ndarray, res: dict) -> None:
        try:
            chat = self._chat()
            if chat is None:
                print("telegram: no chat yet, send /start to the bot first")
                return
            jpeg = cv2.imencode(".jpg", annotate(img, res), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            r = requests.post(API.format(token=self.token, method="sendPhoto"),
                              data={"chat_id": chat, "caption": caption},
                              files={"photo": ("alert.jpg", jpeg, "image/jpeg")}, timeout=20)
            if not r.ok:
                print("telegram:", r.status_code, r.text[:200])
        except requests.RequestException as e:
            print("telegram:", e)


def annotate(img: np.ndarray, res: dict) -> np.ndarray:
    out = img.copy()
    for b in (res.get("fire") or {}).get("boxes", []):
        x1, y1, x2, y2 = b["box"]
        color = (26, 106, 255) if b["label"] == "flame_visible" else (180, 180, 180)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.putText(out, f"{'fire' if b['label'] == 'flame_visible' else 'smoke'} {b['conf']:.0%}",
                    (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    for p in res.get("people", []):
        if p["fall_score"] >= FALL_NOTIFY:
            x1, y1, x2, y2 = p["box"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (94, 63, 244), 3)
            cv2.putText(out, f"fall {p['fall_score']:.0%}", (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (94, 63, 244), 2)
    return out


class StreamAlerts:
    """Per-connection confirmation: notify only after CONFIRM_FRAMES consecutive detections."""

    def __init__(self, tg: Telegram) -> None:
        self.tg = tg
        self.fire_hits = self.fall_hits = 0

    def check(self, img: np.ndarray, res: dict) -> None:
        fire = res.get("fire")
        if fire is None:
            self.fire_hits = 0
        elif not fire.get("reused"):  # a reused result is not a new confirmation
            self.fire_hits = self.fire_hits + 1 if fire["fire_alarm"] else 0
        fallen = [p["fall_score"] for p in res.get("people", []) if p["fall_score"] >= FALL_NOTIFY]
        self.fall_hits = self.fall_hits + 1 if fallen else 0
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if self.fire_hits >= CONFIRM_FRAMES:
            self.tg.send("fire", f"🔥 ตรวจพบไฟไหม้ {fire['flame_conf']:.0%}\n{stamp}", img, res)
        if self.fall_hits >= CONFIRM_FRAMES:
            self.tg.send("fall", f"🚨 ตรวจพบคนล้ม {max(fallen):.0%}\n{stamp}", img, res)
