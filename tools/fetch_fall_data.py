"""Fetch fall-detection training data from Hugging Face.

- DeZan/fall-detection train split (YOLO labels: 0 = fallen)
- Simuletic CCTV lying-down set (every image contains a lying person)
- URFD frames from minhy112/fall-detection-data: end of fall sequences = on the floor,
  start of fall sequences and ADL sequences = not fallen
"""
import collections
import json
import os
import subprocess
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

OUT = os.path.join(os.path.dirname(__file__), "..", "testsets")


def files(ds):
    out = subprocess.run(["curl", "-s", f"https://huggingface.co/api/datasets/{ds}"], capture_output=True).stdout
    return [x["rfilename"] for x in json.loads(out)["siblings"]]


def dl(job):
    ds, f, dst = job
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(["curl", "-sfL", "-o", dst,
                    f"https://huggingface.co/datasets/{ds}/resolve/main/{urllib.parse.quote(f)}"])


jobs = []
for f in files("DeZan/fall-detection"):
    if "/train/" in f:
        jobs.append(("DeZan/fall-detection", f, f"{OUT}/dztrain/{f.split('/')[-2]}/{os.path.basename(f)}"))

ds = "Simuletic/CCTV_Incident_Dataset_Fall_Lying_Down_Detection"
for f in files(ds):
    if "/images/" in f:
        jobs.append((ds, f, f"{OUT}/simuletic/{os.path.basename(f)}"))

seqs = collections.defaultdict(list)
for f in files("minhy112/fall-detection-data"):
    if f.startswith("raw/URFD/") and f.endswith(".png"):
        seqs[f.rsplit("/", 1)[0]].append(f)
for s, fs in seqs.items():
    fs.sort()
    n = len(fs)
    if "/fall/" in s:
        pick = [("pos", q) for q in (0.86, 0.9, 0.94, 0.98)] + [("neg", q) for q in (0.02, 0.12)]
    else:
        pick = [("neg", q) for q in (0.1, 0.4, 0.7, 0.95)]
    for lab, q in pick:
        f = fs[int(n * q)]
        jobs.append(("minhy112/fall-detection-data", f, f"{OUT}/urfd/{lab}/{os.path.basename(f)}"))

print(len(jobs), "files")
with ThreadPoolExecutor(16) as ex:
    list(ex.map(dl, jobs))
