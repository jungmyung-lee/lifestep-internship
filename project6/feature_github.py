"""
Author: Jungmyung Lee

This script defines feature extraction utilities for pose-based basketball
shooting analysis.

Given a shooting video, the code uses YOLOv8-Pose to extract 2D body keypoints
from each frame and computes simple biomechanical features such as elbow joint
angles and normalized wrist height.

The extracted time-series signals are resampled to a fixed length and
z-score normalized, then concatenated into a single feature vector for
downstream machine learning or deep learning models.
"""

# feature.py
import cv2
import numpy as np
import torch

# =====================================================
# Configuration
# =====================================================
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

R_SHOULDER, R_ELBOW, R_WRIST, R_HIP = 6, 8, 10, 12
RESAMPLE_LEN = 80
MIN_VALID_FRAMES = 15   # minimum number of valid frames for stability (dataset size ~100)


# =====================================================
# Math utilities
# =====================================================
def angle_2d(a, b, c):
    """
    Compute a 2D joint angle (ABC), where B is the vertex.
    """
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cosv = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return np.arccos(cosv)


def resample_1d(x, L):
    """
    Resample a 1D time series to length L using linear interpolation.
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 2:
        return None
    src = np.linspace(0, 1, len(x))
    dst = np.linspace(0, 1, L)
    return np.interp(dst, src, x)


def zscore(x):
    """
    Apply z-score normalization.
    """
    std = np.std(x)
    if std < 1e-6:
        return np.zeros_like(x)
    return (x - np.mean(x)) / (std + 1e-6)


# =====================================================
# Video → feature extraction (240-dim)
# =====================================================
def video_to_feature(video_path, model):
    cap = cv2.VideoCapture(video_path)

    elbow_angles = []
    wrist_rel_y  = []
    hip_dummy    = []

    valid_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, device=DEVICE, verbose=False)

        # Skip frames with no detected person or keypoints
        if (
            results is None
            or len(results) == 0
            or results[0].keypoints is None
            or results[0].keypoints.xy is None
            or results[0].keypoints.xy.shape[0] == 0
        ):
            continue

        kp = results[0].keypoints.xy[0].cpu().numpy()

        s = kp[R_SHOULDER]
        e = kp[R_ELBOW]
        w = kp[R_WRIST]
        h = kp[R_HIP]

        # Elbow joint angle
        elbow_angles.append(angle_2d(s, e, w))

        # Relative wrist height (camera-distance normalization)
        shoulder_to_hip = np.linalg.norm(s - h) + 1e-6
        wrist_rel_y.append((w[1] - h[1]) / shoulder_to_hip)

        # Reference signal (kept for alignment)
        hip_dummy.append(0.0)

        valid_frames += 1

    cap.release()

    # =================================================
    # Basic stability check
    # =================================================
    if valid_frames < MIN_VALID_FRAMES:
        return None

    elbow_r = resample_1d(elbow_angles, RESAMPLE_LEN)
    wrist_r = resample_1d(wrist_rel_y, RESAMPLE_LEN)
    hip_r   = resample_1d(hip_dummy, RESAMPLE_LEN)

    if elbow_r is None or wrist_r is None:
        return None

    elbow_r = zscore(elbow_r)
    wrist_r = zscore(wrist_r)
    hip_r   = zscore(hip_r)

    # Final feature vector (240,)
    return np.concatenate([elbow_r, wrist_r, hip_r])
