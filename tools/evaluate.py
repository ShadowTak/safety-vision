"""End-to-end accuracy check of the real pipeline (same code the web app runs).

Usage: .venv/bin/python tools/evaluate.py
Needs: testsets/ (tools/fetch_fall_data.py + fire sets) and samples/.
"""
import glob
import os
import sys

import cv2

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from pipeline import Pipeline  # noqa: E402

T = f"{ROOT}/testsets"
p = Pipeline()


def prf(rows):
    tp = sum(g and q for g, q in rows); fp = sum(q and not g for g, q in rows)
    fn = sum(g and not q for g, q in rows); tn = len(rows) - tp - fp - fn
    return f"accuracy={(tp + tn) / len(rows):.2f} precision={tp / max(tp + fp, 1):.2f} recall={tp / max(tp + fn, 1):.2f} (n={len(rows)})"


# ---- fire: D-Fire labeled sample (0 smoke, 1 fire, empty = no fire) + hard negatives
fire, smoke = [], []
for f in sorted(glob.glob(f"{T}/dfire/*.jpg")):
    cls = {l.split()[0] for l in open(f[:-4] + ".txt") if l.strip()}
    r = p.detect_fire(cv2.imread(f))
    fire.append(("1" in cls, r["fire_alarm"]))
    smoke.append(("0" in cls, r["smoke_alarm"]))
neg = [p.detect_fire(cv2.imread(f)) for f in sorted(glob.glob(f"{T}/fireneg/*.jpg"))]
print("FIRE  (D-Fire)  ", prf(fire))
print("SMOKE (D-Fire)  ", prf(smoke))
print(f"FIRE/SMOKE false alarms on {len(neg)} hard negatives (sunsets, orange lights, dust...): "
      f"{sum(r['fire_alarm'] or r['smoke_alarm'] for r in neg) / len(neg):.0%}")

# ---- fall: DeZan val (held out from training; label 0 = fallen)
rows = []
for f in sorted(glob.glob(f"{T}/falldz/images/*")):
    lab = os.path.splitext(f.replace("/images/", "/labels/"))[0] + ".txt"
    gt = any(l.split()[0] == "0" for l in open(lab) if l.strip())
    rows.append((gt, any(x["fallen"] for x in p.detect_fall(cv2.imread(f), None))))
print("FALL  (DeZan val)", prf(rows))
ok = [f for f in glob.glob(f"{ROOT}/samples/face_*.jpg") + glob.glob(f"{ROOT}/samples/person_standing_*.jpg")]
fa = sum(any(x["fallen"] for x in p.detect_fall(cv2.imread(f), None)) for f in ok)
print(f"FALL  false alarms on {len(ok)} demo photos of people not falling: {fa}")
