import numpy as np

def normalize_landmarks(flat):
    arr = np.asarray(flat, dtype=np.float32)
    single = arr.ndim == 1
    pts = arr.reshape(-1, 21, 3).copy()
    pts -= pts[:, 0:1, :]
    scale = np.linalg.norm(pts[:, 9, :2], axis=1)
    pts /= (scale[:, None, None] + 1e-6)
    out = pts.reshape(len(pts), -1)
    return out[0] if single else out