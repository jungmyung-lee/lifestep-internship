"""
Author: Jungmyung Lee

This script performs single-video inference for pose-based basketball
shooting analysis.

Given an input video, it extracts pose-based features using the feature
extraction module, predicts a shooting score using a trained XGBoost
regression model, and applies SHAP to estimate the relative contribution
of different body components (elbow, wrist, lower body).

The output includes a numeric score along with interpretable, component-
level feedback intended for qualitative shooting form analysis.
"""


# infer_single.py
import numpy as np
import shap
from ultralytics import YOLO
from xgboost import XGBRegressor

from feature import video_to_feature, RESAMPLE_LEN

_yolo = None
_model = None
_explainer = None
_background = None


# =====================================================
# Model loading (lazy loading)
# =====================================================
def load_models():
    """Lazy-load to avoid re-loading on every run."""
    global _yolo, _model, _explainer, _background

    if _yolo is None:
        _yolo = YOLO("yolov8n-pose.pt")

    if _model is None:
        _model = XGBRegressor()
        _model.load_model("model/xgb_regressor.json")

    if _background is None:
        _background = np.load("model/shap_background.npy")

    if _explainer is None:
        # Tree-based model → TreeExplainer
        _explainer = shap.TreeExplainer(_model, _background)

    return _yolo, _model, _explainer


# =====================================================
# Feedback interpretation logic
# =====================================================
def interpret(elbow: float, wrist: float, hip: float) -> list[str]:
    """
    Convert SHAP contribution magnitudes into coach-style feedback.
    Uses relative importance, not absolute judgments.
    """
    msgs = []

    parts = {
        "elbow": elbow,
        "wrist": wrist,
        "lower-body": hip
    }

    # Normalization for relative comparison
    total = sum(parts.values()) + 1e-6
    ratios = {k: v / total for k, v in parts.items()}

    # Sort by contribution magnitude
    ordered = sorted(ratios.items(), key=lambda x: x[1], reverse=True)

    top, mid, low = ordered

    # Primary feedback (most influential factor)
    if top[0] == "elbow":
        msgs.append(
            "Elbow mechanics have the strongest influence on your score. "
            "Focus on keeping the elbow path more consistent throughout the shot."
        )
    elif top[0] == "wrist":
        msgs.append(
            "Wrist control and release trajectory have the strongest influence on your score. "
            "Focus on a smoother, more repeatable release."
        )
    else:
        msgs.append(
            "Lower-body stability has the strongest influence on your score. "
            "Focus on balance and consistent leg drive during the shooting motion."
        )

    # Secondary feedback
    if mid[1] > 0.25:
        if mid[0] == "elbow":
            msgs.append("Elbow consistency also plays a noticeable role—avoid excessive elbow drift or flare.")
        elif mid[0] == "wrist":
            msgs.append("Wrist consistency also matters—pay attention to release timing and follow-through.")
        else:
            msgs.append("Lower-body stability also contributes—maintain a stable base during the shot.")

    # Positive feedback (least influential factor)
    if low[1] < 0.2:
        if low[0] == "elbow":
            msgs.append("Elbow mechanics appear relatively stable compared to other factors.")
        elif low[0] == "wrist":
            msgs.append("Wrist control appears relatively stable compared to other factors.")
        else:
            msgs.append("Lower-body stability appears relatively stable compared to other factors.")

    return msgs


# =====================================================
# Single-video inference
# =====================================================
def infer(video_path: str) -> dict:
    """
    Returns:
    {
      "score": float,
      "confidence": float,
      "details": {"elbow": float, "wrist": float, "hip": float},
      "feedback": [str, ...]
    }
    """
    yolo, model, explainer = load_models()

    feat = video_to_feature(video_path, yolo)
    if feat is None:
        return {"error": "No player detected or insufficient valid frames."}

    X = feat.reshape(1, -1)  # (1, 240)

    # -------------------------------------------------
    # 1) Score prediction
    # -------------------------------------------------
    raw_score = float(model.predict(X)[0])

    # Score stabilization (assumed 0–100 range)
    score = float(np.clip(raw_score, 0, 100))

    # -------------------------------------------------
    # 2) SHAP-based explanation
    # -------------------------------------------------
    shap_vals = explainer.shap_values(X)
    shap_vals = np.array(shap_vals).reshape(-1)  # (240,)

    elbow_vals = np.abs(shap_vals[:RESAMPLE_LEN])
    wrist_vals = np.abs(shap_vals[RESAMPLE_LEN:2 * RESAMPLE_LEN])
    hip_vals   = np.abs(shap_vals[2 * RESAMPLE_LEN:3 * RESAMPLE_LEN])

    elbow = float(np.mean(elbow_vals))
    wrist = float(np.mean(wrist_vals))
    hip   = float(np.mean(hip_vals))

    # -------------------------------------------------
    # 3) Confidence estimation (based on explanation stability)
    # -------------------------------------------------
    total_contrib = elbow + wrist + hip
    confidence = float(np.clip(total_contrib / (total_contrib + 1.0), 0.3, 0.9))

    feedback = interpret(elbow, wrist, hip)

    return {
        "score": round(score, 1),
        "confidence": round(confidence, 2),
        "details": {
            "elbow": round(elbow, 4),
            "wrist": round(wrist, 4),
            "hip": round(hip, 4)
        },
        "feedback": feedback
    }
