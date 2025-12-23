"""
Author: Jungmyung Lee

This script implements a pose-based basketball shooting analysis pipeline
developed independently during my LifeStep internship.

The system uses YOLOv8-Pose to extract 2D body keypoints from shooting videos,
derives simple biomechanical features (e.g., elbow angle and normalized wrist height),
and models the temporal motion pattern using a CNN for local feature extraction
followed by an LSTM to capture sequential dynamics.

Each shooting sequence is resampled to a fixed length and normalized to enable
stable learning. The final model performs binary classification of shooting form
quality (GOOD / BAD) using stratified K-fold cross-validation.
"""


import os
import glob
import cv2
import numpy as np

import torch
import torch.nn as nn

from ultralytics import YOLO
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split



# =====================================================
# Configuration
# =====================================================
DEVICE = "cpu"
MODEL_PATH = "yolov8n-pose.pt"

DATA_ROOT = "data_clips"
GOOD_DIR = os.path.join(DATA_ROOT, "good")
BAD_DIR  = os.path.join(DATA_ROOT, "bad")

RESAMPLE_LEN = 80
EPOCHS = 80
LR = 1e-3
PATIENCE = 8
N_SPLITS = 5

# YOLOv8 COCO keypoint indices
R_SHOULDER = 6
R_ELBOW    = 8
R_WRIST    = 10
R_HIP      = 12


# =====================================================
# Utility functions
# =====================================================
def safe_first_person(results):
    """
    Safely extract the first detected person's keypoints.
    Returns None if no valid person is detected.
    """
    if results is None or len(results) == 0:
        return None
    kp = results[0].keypoints
    if kp is None or kp.xy is None or kp.xy.shape[0] == 0:
        return None
    return kp.xy[0].cpu().numpy()  # (17, 2)


def angle_2d(a, b, c):
    """
    Compute a 2D joint angle given three points (a-b-c).
    """
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cosv = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return np.arccos(cosv)


def resample_1d(x, target_len):
    """
    Linearly resample a 1D signal to a fixed length.
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 2:
        return None
    src = np.linspace(0, 1, len(x))
    dst = np.linspace(0, 1, target_len)
    return np.interp(dst, src, x)


def zscore(x):
    """
    Apply z-score normalization.
    """
    return (x - np.mean(x)) / (np.std(x) + 1e-6)


# =====================================================
# Video → temporal feature extraction (80, 3)
# =====================================================
def video_to_feature(video_path, model):
    """
    Convert a shooting video into a fixed-length time-series feature.
    Output shape: (80, 3)
    """
    cap = cv2.VideoCapture(video_path)

    elbow, wrist_y, hip_y = [], [], []
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
        elbow.append(angle_2d(s, e, w))

        # Relative wrist height normalized by torso length
        shoulder_to_hip = np.linalg.norm(s - h)
        wrist_y.append((w[1] - h[1]) / (shoulder_to_hip + 1e-6))

        # Initialize reference hip position (first valid frame)
        if hip0 is None:
            hip0 = h[1]
            torso0 = np.linalg.norm(s - h) + 1e-6
        
        # Normalized hip vertical displacement (baseline signal)
        hip_disp = (h[1] - hip0) / torso0
        hip_y.append(hip_disp)


    cap.release()

    if len(elbow) < 10:
        return None

    elbow = zscore(resample_1d(elbow, RESAMPLE_LEN))
    wrist = zscore(resample_1d(wrist_y, RESAMPLE_LEN))
    hip   = zscore(resample_1d(hip_y, RESAMPLE_LEN))

    return np.stack([elbow, wrist, hip], axis=1)  # (80, 3)


# =====================================================
# Dataset construction
# =====================================================
def build_dataset(model):
    """
    Build feature tensors and labels from GOOD / BAD video folders.
    """
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
# CNN + LSTM model
# =====================================================
class CNN_LSTM(nn.Module):
    """
    CNN extracts local temporal patterns,
    LSTM models long-range temporal dynamics.
    """
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)      # (B, C, T)
        x = self.conv(x)
        x = x.permute(0, 2, 1)      # (B, T, C)

        _, (h, _) = self.lstm(x)
        return self.fc(self.dropout(h[-1]))


# =====================================================
# Training (K-Fold Cross Validation + Early Stopping)
# =====================================================
def train_kfold(X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    accs, f1s = [], []

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        print(f"\n========== Fold {fold} ==========")

        model = CNN_LSTM().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss()

        # Train / Validation split inside training fold
        X_train, X_val, y_train, y_val = train_test_split(
            X[tr], y[tr],
            test_size=0.2,
            stratify=y[tr],
            random_state=42
        )
        
        Xtr = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        ytr = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
        
        Xval = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        yval = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
        
        Xte = torch.tensor(X[te], dtype=torch.float32).to(DEVICE)
        yte = torch.tensor(y[te], dtype=torch.float32).to(DEVICE)


        best_loss = float("inf")
        best_state = None
        patience_cnt = 0
        
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
        
            logits = model(Xtr).view(-1)
            loss = criterion(logits, ytr)
            loss.backward()
            optimizer.step()
        
            # Validation (NOT test)
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(Xval).view(-1), yval).item()
        
            if val_loss < best_loss:
                best_loss = val_loss
                patience_cnt = 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        model.to(DEVICE)


        # -------- Evaluation --------
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(Xte).view(-1)).cpu().numpy()

        preds = (probs > 0.5).astype(int)
        y_true = yte.cpu().numpy()

        acc = accuracy_score(y_true, preds)
        f1  = f1_score(y_true, preds)

        accs.append(acc)
        f1s.append(f1)

        print("Accuracy:", acc)
        print("F1-score:", f1)
        print(confusion_matrix(y_true, preds))

    print("\n===== Cross-Validation Result =====")
    print(f"Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"F1-score: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    yolo = YOLO(MODEL_PATH)

    print("Building dataset...")
    X, y = build_dataset(yolo)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if len(np.unique(y)) < 2:
        raise RuntimeError("Both GOOD and BAD samples are required")

    train_kfold(X, y)
