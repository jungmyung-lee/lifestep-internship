"""
Author: Jungmyung Lee

This script trains and evaluates a pose-based classification model for
basketball shooting form analysis.

It extracts pose-based feature vectors from shooting videos using YOLOv8-Pose,
labels them as GOOD or BAD, and trains an XGBoost classifier to distinguish
between the two classes.

Model performance is evaluated using stratified K-fold cross-validation,
and results are reported using accuracy, F1-score, confusion matrix,
and a classification report.
"""


import os
import glob
import cv2
import numpy as np

from ultralytics import YOLO
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# =====================================================
# Configuration
# =====================================================
DEVICE = "mps"  # Mac M1–M4 (fallback to "cpu" if unavailable)
MODEL_PATH = "yolov8n-pose.pt"

DATA_ROOT = "data_clips"
GOOD_DIR = os.path.join(DATA_ROOT, "good")
BAD_DIR  = os.path.join(DATA_ROOT, "bad")

RESAMPLE_LEN = 80  # fixed length for temporal signals

# YOLOv8 COCO keypoints
R_SHOULDER = 6
R_ELBOW    = 8
R_WRIST    = 10
R_HIP      = 12

# =====================================================
# Utility functions
# =====================================================
def safe_first_person(results):
    if results is None or len(results) == 0:
        return None
    kp = results[0].keypoints
    if kp is None or kp.xy is None or kp.xy.shape[0] == 0:
        return None
    return kp.xy[0].cpu().numpy()  # (17, 2)


def angle_2d(a, b, c):
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cosv = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return np.arccos(cosv)


def resample_1d(x, target_len):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 2:
        return None
    src = np.linspace(0, 1, len(x))
    dst = np.linspace(0, 1, target_len)
    return np.interp(dst, src, x)


def zscore(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-6)

# =====================================================
# 1️⃣ Video → feature vector (240-dim)
# =====================================================
def video_to_feature(video_path, model):
    cap = cv2.VideoCapture(video_path)

    elbow_angles = []
    wrist_rel_y  = []
    hip_y        = []

    hip0 = None
    torso0 = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, device=DEVICE, verbose=False)
        kp = safe_first_person(results)
        if kp is None:
            continue

        s = kp[R_SHOULDER]
        e = kp[R_ELBOW]
        w = kp[R_WRIST]
        h = kp[R_HIP]

        # Elbow joint angle
        elbow_angles.append(angle_2d(s, e, w))

        # Wrist height (normalized)
        shoulder_to_hip = np.linalg.norm(s - h) + 1e-6
        wrist_rel_y.append((w[1] - h[1]) / shoulder_to_hip)

        # Initialize hip reference
        if hip0 is None:
            hip0 = h[1]
            torso0 = shoulder_to_hip

        # Normalized hip vertical displacement (baseline)
        hip_disp = (h[1] - hip0) / torso0
        hip_y.append(hip_disp)

    cap.release()

    if len(elbow_angles) < 10:
        return None

    elbow_r = zscore(resample_1d(elbow_angles, RESAMPLE_LEN))
    wrist_r = zscore(resample_1d(wrist_rel_y, RESAMPLE_LEN))
    hip_r   = zscore(resample_1d(hip_y, RESAMPLE_LEN))

    return np.concatenate([elbow_r, wrist_r, hip_r])  # (240,)


# =====================================================
# 2️⃣ Dataset construction
# =====================================================
def build_dataset(model):
    X, y = [], []

    for folder, label in [(GOOD_DIR, 1), (BAD_DIR, 0)]:
        for vp in glob.glob(folder + "/*"):
            feat = video_to_feature(vp, model)
            if feat is not None:
                X.append(feat)
                y.append(label)
                print(f"[OK] {os.path.basename(vp)}")

    return np.array(X), np.array(y)

# =====================================================
# 3️⃣ XGBoost training & evaluation (5-Fold)
# =====================================================
def train_xgboost(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds_all = np.zeros_like(y)

    accs, f1s = [], []

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        clf = XGBClassifier(
            n_estimators=400,
            max_depth=3,              # regularization for small datasets
            learning_rate=0.02,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=4,
            random_state=42
        )

        clf.fit(X[tr], y[tr])
        probs = clf.predict_proba(X[te])[:, 1]
        preds = (probs > 0.5).astype(int)
        preds_all[te] = preds

        acc = accuracy_score(y[te], preds)
        f1  = f1_score(y[te], preds)

        accs.append(acc)
        f1s.append(f1)

        print(f"Fold {fold} | Accuracy: {acc:.3f} | F1: {f1:.3f}")

    print("\n===== Cross-Validation Result =====")
    print(f"Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"F1-score: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    print("\nConfusion Matrix (All Data)")
    print(confusion_matrix(y, preds_all))

    print("\nClassification Report")
    print(classification_report(y, preds_all, target_names=["BAD", "GOOD"]))

# =====================================================
# main
# =====================================================
if __name__ == "__main__":
    model = YOLO(MODEL_PATH)

    print("Building dataset...")
    X, y = build_dataset(model)

    print("X shape:", X.shape)  # (N, 240)
    print("y shape:", y.shape)

    if len(np.unique(y)) < 2:
        raise RuntimeError("Both GOOD and BAD samples are required")

    train_xgboost(X, y)
