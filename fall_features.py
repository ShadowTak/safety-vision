"""Scale/position-invariant pose features for the fall classifier (shared by training and inference)."""
import numpy as np

L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12


def features(box, kp: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """box = x1,y1,x2,y2; kp = (17,3) pixel x, y, conf."""
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    conf = kp[:, 2]
    vis = conf > 0.3
    # keypoints in box frame (0..1)
    kb = np.stack([(kp[:, 0] - x1) / bw, (kp[:, 1] - y1) / bh], 1)
    # keypoints relative to body centre, scaled by box diagonal (keeps orientation)
    c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    kc = (kp[:, :2] - c) / np.hypot(bw, bh)
    kb[~vis] = -1
    kc[~vis] = 0
    sh = (kp[L_SH, :2] + kp[R_SH, :2]) / 2
    hip = (kp[L_HIP, :2] + kp[R_HIP, :2]) / 2
    torso_ok = vis[[L_SH, R_SH, L_HIP, R_HIP]].all()
    v = sh - hip
    angle = np.degrees(np.arctan2(abs(v[0]), abs(v[1]) + 1e-6)) if torso_ok else -1
    up = -v[1] / (np.hypot(*v) + 1e-6) if torso_ok else 0  # 1 = head up, -1 = upside down
    ys = kp[vis, 1]
    xs = kp[vis, 0]
    spread = ((ys.max() - ys.min()) / (xs.max() - xs.min() + 1e-6)) if vis.sum() >= 3 else -1
    g = [bw / bh, angle, up, spread, vis.mean(), (y2 / img_h), bh / img_h, bw / img_w]
    return np.concatenate([kb.ravel(), kc.ravel(), conf, g]).astype(np.float32)
