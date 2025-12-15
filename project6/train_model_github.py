"""
Author: Jungmyung Lee

This script trains a regression model for pose-based basketball
shooting analysis.

It loads shooting videos and corresponding numeric form scores,
extracts pose-based feature vectors using the feature extraction module,
and trains an XGBoost regression model.

Model performance is evaluated using K-fold cross-validation, and a final
model is trained on the full dataset and saved for downstream inference
and interpretability analysis.
"""


# train_model.py
import os
import csv
import numpy as np

from ultralytics import YOLO
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

from feature import video_to_feature

# =====================================================
# Configuration
# =====================================================
VIDEO_DIR = "videos"
LABEL_FILE = "labels.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

N_SPLITS = 5           # dataset size ~107 → suitable for 5-fold CV
RANDOM_STATE = 42

# =====================================================
# Data loading
# =====================================================
yolo = YOLO("yolov8n-pose.pt")

X, y = [], []

with open(LABEL_FILE) as f:
    for r in csv.DictReader(f):
        video_path = os.path.join(VIDEO_DIR, r["video_name"])
        feat = video_to_feature(video_path, yolo)

        if feat is not None:
            X.append(feat)
            y.append(float(r["form_score"]))

X = np.array(X)
y = np.array(y)

# Safety check (flatten if needed)
if X.ndim != 2:
    X = X.reshape(X.shape[0], -1)

print("Dataset shape:", X.shape)
print("Label shape:", y.shape)

# =====================================================
# K-Fold training & evaluation
# =====================================================
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

maes, r2s = [], []

for fold, (tr, te) in enumerate(kf.split(X), 1):
    print(f"\n========== Fold {fold} ==========")

    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE
    )

    model.fit(X[tr], y[tr])

    preds = model.predict(X[te])
    mae = mean_absolute_error(y[te], preds)
    r2  = r2_score(y[te], preds)

    maes.append(mae)
    r2s.append(r2)

    print(f"MAE: {mae:.3f}")
    print(f"R² : {r2:.3f}")

print("\n===== Cross-Validation Result =====")
print(f"MAE: {np.mean(maes):.3f} ± {np.std(maes):.3f}")
print(f"R² : {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")

# =====================================================
# Train final model on full dataset
# =====================================================
final_model = XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=RANDOM_STATE
)

final_model.fit(X, y)

# =====================================================
# Save model & SHAP background
# =====================================================
final_model.save_model(os.path.join(MODEL_DIR, "xgb_regressor.json"))
np.save(os.path.join(MODEL_DIR, "shap_background.npy"), X)

print("\n✅ Final model trained & saved")
