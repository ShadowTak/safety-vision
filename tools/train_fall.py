"""Train the pose-based fall classifier and evaluate it on the held-out DeZan val set.

Usage: .venv/bin/python tools/train_fall.py
Writes models/fall_clf.joblib
"""
import glob
import os
import sys

import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from ultralytics import YOLO

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from fall_features import features  # noqa: E402

T = f"{ROOT}/testsets"
pose = YOLO(f"{ROOT}/yolo11m-pose.pt")


def people(path):
    img = cv2.imread(path)
    if img is None:
        return [], None
    r = pose.predict(img, device="mps", imgsz=640, conf=0.25, verbose=False)[0]
    out = []
    if r.keypoints is not None:
        for b, kp in zip(r.boxes.xyxy.cpu().numpy(), r.keypoints.data.cpu().numpy()):
            out.append((b, kp))
    return out, img.shape


def yolo_boxes(lab, w, h):
    out = []
    for line in open(lab):
        p = line.split()
        if len(p) >= 5:
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
            out.append((c, [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-6)


def from_labeled(img_glob, lab_fn, pos_cls):
    """Persons matched (IoU>=0.3) to a GT box; label = GT class is a fall class."""
    X, y, groups = [], [], []
    for f in sorted(glob.glob(img_glob)):
        lab = lab_fn(f)
        if not os.path.exists(lab):
            continue
        ppl, shp = people(f)
        if not ppl:
            continue
        gts = yolo_boxes(lab, shp[1], shp[0])
        for b, kp in ppl:
            best = max(gts, key=lambda g: iou(b, g[1]), default=None)
            if best is None or iou(b, best[1]) < 0.3:
                continue
            X.append(features(b, kp, shp[1], shp[0]))
            y.append(int(best[0] in pos_cls))
            groups.append(f)
    return X, y, groups


def from_folder(img_glob, label):
    """URFD frames: single subject, take the largest detected person."""
    X, y = [], []
    for f in sorted(glob.glob(img_glob)):
        ppl, shp = people(f)
        if not ppl:
            continue
        b, kp = max(ppl, key=lambda p: (p[0][2] - p[0][0]) * (p[0][3] - p[0][1]))
        X.append(features(b, kp, shp[1], shp[0]))
        y.append(label)
    return X, y


def lab_for(f):
    return os.path.splitext(f.replace("/images/", "/labels/"))[0] + ".txt"


Xtr, ytr = [], []
# DeZan train: fetch saved images and labels side by side in dztrain/train
X, y, _ = from_labeled(f"{T}/dztrain/train/*.[jp][pn]g", lambda f: os.path.splitext(f)[0] + ".txt", {0})
Xtr += X; ytr += y; print("dezan train", len(y), sum(y))
X, y, _ = from_labeled(f"{T}/simuletic/*.png", lambda f: f"{T}/simuletic_labels/" + os.path.basename(f)[:-4] + ".txt", {0})
Xtr += X; ytr += y; print("simuletic", len(y), sum(y))
for lab, val in (("pos", 1), ("neg", 0)):
    X, y = from_folder(f"{T}/urfd/{lab}/*.png", val)
    Xtr += X; ytr += y; print("urfd", lab, len(y))

# COCO everyday people (sitting, crouching, reclining, sport) as negatives. People whose
# torso is near-horizontal are skipped: COCO does not say whether they fell or are resting.
n = 0
for f in sorted(glob.glob(f"{T}/coco/imgs/*.jpg")):
    ppl, shp = people(f)
    for b, kp in ppl:
        x = features(b, kp, shp[1], shp[0])
        angle, rel_h = x[-7], x[-2]
        if angle < 50 and rel_h >= 0.2:  # angle -1 = torso not visible (close-ups)
            Xtr.append(x); ytr.append(0); n += 1
print("coco negatives", n)

Xtr, ytr = np.array(Xtr), np.array(ytr)
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                     l2_regularization=1.0, class_weight="balanced", random_state=0).fit(Xtr, ytr)
joblib.dump(clf, f"{ROOT}/models/fall_clf.joblib")
print("trained on", len(ytr), "people,", ytr.sum(), "fallen")

# ---- held-out evaluation (image level, DeZan val) ----
rows = []
for f in sorted(glob.glob(f"{T}/falldz/images/*")):
    gt = any(l.split()[0] == "0" for l in open(lab_for(f)) if l.strip())
    ppl, shp = people(f)
    p = max([clf.predict_proba(features(b, kp, shp[1], shp[0])[None])[0, 1] for b, kp in ppl], default=0)
    rows.append((gt, p))
for t in (0.3, 0.4, 0.5, 0.6, 0.7):
    tp = sum(g and p >= t for g, p in rows); fp = sum((not g) and p >= t for g, p in rows)
    fn = sum(g and p < t for g, p in rows); tn = len(rows) - tp - fp - fn
    print(f"t={t} acc={(tp + tn) / len(rows):.2f} recall={tp / (tp + fn):.2f} FP={fp}/{fp + tn}")
