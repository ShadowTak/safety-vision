"""Image-level benchmark of fire/smoke detectors on D-Fire sample + hard negatives."""
import glob, os, sys, cv2, numpy as np
from ultralytics import YOLO
dev="mps"
data=[]
for f in sorted(glob.glob("testsets/dfire/*.jpg")):
    lab=f[:-4]+".txt"; cls={l.split()[0] for l in open(lab) if l.strip()} if os.path.exists(lab) else set()
    data.append((f,"1" in cls,"0" in cls))
neg=[(f,False,False) for f in sorted(glob.glob("testsets/fireneg/*.jpg"))]
print("dfire",len(data),"fire",sum(d[1] for d in data),"smoke",sum(d[2] for d in data),"neg",sum(not d[1] and not d[2] for d in data),"hardneg",len(neg))
def f1(gt,pr):
    tp=sum(g and p for g,p in zip(gt,pr)); fp=sum(p and not g for g,p in zip(gt,pr)); fn=sum(g and not p for g,p in zip(gt,pr))
    P=tp/max(tp+fp,1); R=tp/max(tp+fn,1); return P,R,2*P*R/max(P+R,1e-9)
for w in sys.argv[1:]:
    try: m=YOLO(w)
    except Exception as e: print(w,"LOAD FAIL",e); continue
    names={k:v.lower() for k,v in m.names.items()}
    fid=[k for k,v in names.items() if "fire" in v or "flame" in v]; sid=[k for k,v in names.items() if "smoke" in v]
    def score(f):
        r=m.predict(f,device=dev,conf=0.05,verbose=False,imgsz=640)[0]
        c=r.boxes.cls.cpu().numpy().astype(int); s=r.boxes.conf.cpu().numpy()
        return max([x for x,k in zip(s,c) if k in fid],default=0), max([x for x,k in zip(s,c) if k in sid],default=0)
    sc=[score(f) for f,*_ in data]; ns=[score(f) for f,*_ in neg]
    best=None
    for t in (0.25,0.35,0.5):
        P,R,F=f1([d[1] for d in data],[s[0]>=t for s in sc])
        sP,sR,sF=f1([d[2] for d in data],[s[1]>=t for s in sc])
        fa=np.mean([max(s)>=t for s in ns])
        print(f"{os.path.basename(w):16s} names={list(names.values())} t={t} FIRE P={P:.2f} R={R:.2f} F1={F:.2f} | SMOKE P={sP:.2f} R={sR:.2f} F1={sF:.2f} | hardneg FA={fa:.2f}")
