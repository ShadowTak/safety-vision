"""Shared detection pipeline: fire/smoke, person fall, facial emotion."""
from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from ultralytics import YOLO

from fall_features import features

MODELS = Path(__file__).parent / "models"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FIRE_ALARM = 0.4  # ensemble score threshold (D-Fire sample: fire P=0.98 R=0.93)
FALL_ALARM = 0.5
FIRE_EVERY = 2  # live: fire changes slowly, so run it every 2nd frame and reuse the result
POSE_IMGSZ = 800  # better on small/distant people than 640 (DeZan val acc 0.90 -> 0.92)
HALF = DEVICE == "mps"  # fp16 on Apple GPU: ~35% faster pose, same accuracy (tools/evaluate.py)

# COCO keypoint indices
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12

EMOTIONS_TH = {
    "Happiness": "ยิ้ม/มีความสุข", "Sadness": "เศร้า", "Anger": "โกรธ", "Fear": "กลัว",
    "Surprise": "ตกใจ", "Disgust": "รังเกียจ", "Neutral": "เฉยๆ", "Contempt": "ดูถูก",
}


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-6)


class _Tracker:
    """Greedy IoU matching of boxes to tracks from earlier frames of one stream."""

    def __init__(self, ttl: float = 1.5) -> None:
        self.tracks: dict[int, dict] = {}  # id -> {"box", "seen", ...per-track data}
        self.ttl, self._next = ttl, 0

    def match(self, boxes: list, now: float) -> list[dict]:
        self.tracks = {k: t for k, t in self.tracks.items() if now - t["seen"] < self.ttl}
        free, out = set(self.tracks), []
        for b in boxes:
            k = max(free, key=lambda k: _iou(b, self.tracks[k]["box"]), default=None)
            if k is None or _iou(b, self.tracks[k]["box"]) < 0.3:
                k, self._next = self._next, self._next + 1
                self.tracks[k] = {}
            free.discard(k)
            t = self.tracks[k]
            t["box"], t["seen"] = b, now
            out.append(t)
        return out


@dataclass
class TrackState:
    """Per-stream temporal state: each person's body-centre history (fall) and each face's
    smoothed emotion probabilities. People and faces are tracked separately because they run
    on different threads."""
    people: _Tracker = field(default_factory=_Tracker)
    faces: _Tracker = field(default_factory=lambda: _Tracker(ttl=1.0))
    frame: int = 0
    fire: dict | None = None  # last fire result, reused between fire runs


class Pipeline:
    def __init__(self) -> None:
        self.fire_models = [YOLO(str(MODELS / "fire_rabah.pt")), YOLO(str(MODELS / "fire_n26.pt"))]
        self.pose = YOLO(str(Path(__file__).parent / "yolo11m-pose.pt"))
        self.fall_clf = joblib.load(MODELS / "fall_clf.joblib")
        self._cpu = ThreadPoolExecutor(max_workers=1)
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        self.emo = HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf")
        self.face = cv2.FaceDetectorYN.create(
            str(MODELS / "face_yunet.onnx"), "", (320, 320), score_threshold=0.4)
        # warm up so first live frame isn't slow
        z = np.zeros((480, 640, 3), np.uint8)
        for m in self.fire_models:
            m.predict(z, device=DEVICE, verbose=False)
        self.pose.predict(z, device=DEVICE, verbose=False)

    # ---------------------------------------------------------------- fire
    def detect_fire(self, img: np.ndarray) -> dict:
        """Two-detector ensemble. Image-level score = mean of each model's best score per class
        (benchmarked in tools/bench_fire.py: fewer false alarms than either model alone)."""
        h, w = img.shape[:2]
        per_model = []  # [(best_flame, best_smoke, boxes)]
        for m in self.fire_models:
            r = m.predict(img, device=DEVICE, imgsz=640, conf=0.05, half=HALF, verbose=False)[0]
            best = {"flame_visible": 0.0, "smoke_visible": 0.0}
            boxes = []
            for xyxy, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                label = "smoke_visible" if "smoke" in m.names[int(k)].lower() else "flame_visible"
                best[label] = max(best[label], c)
                boxes.append((label, c, [int(v) for v in xyxy]))
            per_model.append((best, boxes))
        flame_conf = float(np.mean([b["flame_visible"] for b, _ in per_model]))
        smoke_conf = float(np.mean([b["smoke_visible"] for b, _ in per_model]))
        # boxes to show: confident boxes from both models, merged per class with NMS
        shown, flame_mask, smoke_mask = [], np.zeros((h, w), bool), np.zeros((h, w), bool)
        for label, mask in (("flame_visible", flame_mask), ("smoke_visible", smoke_mask)):
            cand = [(c, bx) for _, boxes in per_model for lb, c, bx in boxes if lb == label and c >= 0.3]
            if not cand:
                continue
            keep = cv2.dnn.NMSBoxes([[x1, y1, x2 - x1, y2 - y1] for _, (x1, y1, x2, y2) in cand],
                                    [c for c, _ in cand], 0.3, 0.45)
            for i in np.array(keep).flatten():
                c, (x1, y1, x2, y2) = cand[i]
                shown.append({"label": label, "conf": round(c, 3), "box": [x1, y1, x2, y2]})
                mask[max(y1, 0):y2, max(x1, 0):x2] = True
        return {
            "boxes": shown,
            "flame_conf": round(flame_conf, 3),
            "smoke_conf": round(smoke_conf, 3),
            "flame_area_pct": round(100 * flame_mask.mean(), 1),
            "smoke_area_pct": round(100 * smoke_mask.mean(), 1),
            "fire_alarm": flame_conf >= FIRE_ALARM,
            "smoke_alarm": smoke_conf >= FIRE_ALARM,
        }

    # ---------------------------------------------------------------- fall
    def detect_fall(self, img: np.ndarray, state: TrackState | None, conf: float = 0.25) -> list[dict]:
        r = self.pose.predict(img, device=DEVICE, imgsz=POSE_IMGSZ, conf=conf, half=HALF, verbose=False)[0]
        h, w = img.shape[:2]
        now = time.time()
        people = []
        if r.keypoints is None or r.boxes is None:
            return people
        kps = r.keypoints.data.cpu().numpy()  # (n,17,3)
        boxes = r.boxes.xyxy.tolist()
        tracks = state.people.match(boxes, now) if state is not None else [None] * len(boxes)
        for (x1, y1, x2, y2), kp, track in zip(boxes, kps, tracks):
            bw, bh = x2 - x1, y2 - y1
            ar = bw / max(bh, 1)
            ok = lambda i: kp[i, 2] > 0.3
            # learned score: pose-feature classifier (tools/train_fall.py)
            score = float(self.fall_clf.predict_proba(features((x1, y1, x2, y2), kp, w, h)[None])[0, 1])
            # human-readable reasons (explanations only, not part of the score)
            reasons, angle = [], None
            if ok(L_SH) and ok(R_SH) and ok(L_HIP) and ok(R_HIP):
                v = (kp[L_SH, :2] + kp[R_SH, :2]) / 2 - (kp[L_HIP, :2] + kp[R_HIP, :2]) / 2
                angle = float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]) + 1e-6)))  # 0=upright, 90=horizontal
                if angle > 55:
                    reasons.append(f"ลำตัวเอียง {angle:.0f}°")
                if v[1] > 0:
                    reasons.append("ศีรษะต่ำกว่าสะโพก")
            if ar > 1.1:
                reasons.append(f"กรอบกว้างกว่าสูง ({ar:.2f})")
            # temporal (live only): rapid drop of the body centre
            falling_motion = False
            cy = (y1 + y2) / 2 / h
            if track is not None:
                hist = track.setdefault("hist", deque(maxlen=45))  # (t, cy_norm)
                hist.append((now, cy))
                old = [s for s in hist if 0.2 < now - s[0] < 1.2]
                if old and cy - old[0][1] > 0.12 and (angle or 0) > 40:
                    falling_motion = True
                    score = max(score, 0.5) + 0.2
                    reasons.append("ล้มลงอย่างรวดเร็ว")
            score = min(score, 1.0)
            people.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "fall_score": round(score, 2),
                "fallen": score >= FALL_ALARM,
                "angle": None if angle is None else round(angle, 1),
                "aspect": round(ar, 2),
                "moving_fall": falling_motion,
                "reasons": reasons,
                "keypoints": [[round(float(x)), round(float(y)), round(float(c), 2)] for x, y, c in kp],
            })
        return people

    # ---------------------------------------------------------------- face
    def detect_emotion(self, img: np.ndarray, state: TrackState | None = None) -> list[dict]:
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            img_s = cv2.resize(img, None, fx=scale, fy=scale)
        else:
            img_s = img
        self.face.setInputSize((img_s.shape[1], img_s.shape[0]))
        _, faces = self.face.detect(img_s)
        out = []
        if faces is None:
            return out
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = [(f[:4] / scale).tolist() for f in faces]
        boxes = [[x, y, x + fw, y + fh] for x, y, fw, fh in boxes]
        tracks = state.faces.match(boxes, time.time()) if state is not None else [None] * len(boxes)
        for f, track in zip(faces, tracks):
            x, y, fw, fh = (f[:4] / scale).astype(int)
            pad = int(0.1 * max(fw, fh))
            x0, y0 = max(x - pad, 0), max(y - pad, 0)
            x1, y1 = min(x + fw + pad, w), min(y + fh + pad, h)
            crop = rgb[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            _, scores = self.emo.predict_emotions(crop, logits=False)
            if track is not None:  # live: smooth over frames so the label does not flicker
                prev = track.get("probs")
                scores = scores if prev is None else 0.5 * prev + 0.5 * scores
                track["probs"] = scores
            names = list(self.emo.idx_to_class.values())
            probs = {n: float(s) for n, s in zip(names, scores)}
            emo = names[int(np.argmax(scores))]
            sad = probs.get("Sadness", 0.0)
            # "crying" proxy: strong sadness with fear/anger/disgust mixed in (distressed face)
            distress = sad + 0.5 * (probs.get("Fear", 0) + probs.get("Disgust", 0))
            if emo == "Sadness" and sad > 0.55 and distress > 0.7:
                label = "Crying"
            else:
                label = emo
            out.append({
                "box": [int(x0), int(y0), int(x1), int(y1)],
                "emotion": label,
                "emotion_th": "ร้องไห้" if label == "Crying" else EMOTIONS_TH.get(label, label),
                "conf": round(probs[emo], 3),
                "probs": {k: round(v, 3) for k, v in probs.items()},
                "face_score": round(float(f[14]), 3),
            })
        return out

    # ---------------------------------------------------------------- all
    def analyze(self, img: np.ndarray, state: TrackState | None = None,
                fire: bool = True, fall: bool = True, emotion: bool = True) -> dict:
        t0 = time.perf_counter()
        res: dict = {"w": img.shape[1], "h": img.shape[0]}
        # emotion runs on CPU (onnxruntime) while fire + fall use the GPU (MPS)
        faces = self._cpu.submit(self.detect_emotion, img, state) if emotion else None
        if fire:
            if state is None or state.fire is None or state.frame % FIRE_EVERY == 0:
                res["fire"] = self.detect_fire(img)
                if state is not None:
                    state.fire = res["fire"]
            else:
                res["fire"] = {**state.fire, "reused": True}
        elif state is not None:
            state.fire = None  # never reuse a result from before fire was switched off
        if state is not None:
            state.frame += 1
        if fall:
            res["people"] = self.detect_fall(img, state)
        if faces:
            res["faces"] = faces.result()
        res["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return res
