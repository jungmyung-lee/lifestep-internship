# AI Basketball Shooting Coach  
### Pose-Based Shooting Form Scoring with XGBoost Regression + SHAP

---

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Dataset and Data Curation](#dataset-and-data-curation)
- [Input Data Specification](#input-data-specification)
- [Pose Estimation with YOLOv8-Pose](#pose-estimation-with-yolov8-pose)
- [Biomechanics-Informed Feature Design](#biomechanics-informed-feature-design)
- [Temporal Window Definition for Feature Extraction](#temporal-window-definition-for-feature-extraction)
- [Model: XGBoost Regression](#model-xgboost-regression)
  - [Why Regression Instead of Classification](#why-regression-instead-of-classification)
  - [Feature Vector Construction](#feature-vector-construction)
  - [XGBoost Architecture and Hyperparameters](#xgboost-architecture-and-hyperparameters)
  - [Why XGBoost Works Well in This Project](#why-xgboost-works-well-in-this-project)
- [Training and Evaluation Protocol](#training-and-evaluation-protocol)
- [Explainability with SHAP](#explainability-with-shap)
- [AI Coach Feedback Design](#ai-coach-feedback-design)
- [Experimental Results](#experimental-results)
- [Comparison with Project 5](#comparison-with-project-5)
- [Limitations](#limitations)
- [Conclusion](#conclusion)
- [Technologies Used](#technologies-used)
- [Author](#author)

---

## Project Overview

This project implements an **AI Basketball Shooting Coach** that predicts a
**continuous shooting-form score (0–100)** from a single shooting video and provides
**coach-style, body-part-level feedback**.

The system is built upon a **pose-based shooting analysis pipeline**
and uses:

- YOLOv8-Pose for 2D human pose estimation  
- domain-informed biomechanical feature extraction  
- XGBoost regression for score prediction  
- SHAP for model explainability  

A lightweight **Streamlit web interface** allows users to upload a shooting video
and receive an interpretable score along with actionable feedback.

---

## Motivation

Basketball shooting quality is traditionally evaluated subjectively by coaches.
However, subtle differences in elbow mechanics, wrist release, and lower-body stability
are difficult to quantify consistently.

Drawing on **over four years of basketball coaching experience**, this project aims to:

- formalize shooting-form evaluation as a numeric regression problem  
- preserve biomechanical interpretability  
- operate reliably under **small-data conditions**  

Rather than building a black-box video model,
the system intentionally emphasizes **domain-aligned feature design and explainability**.

---

## Dataset and Data Curation

- Data source: real basketball class session videos  
- Total samples: **107 shooting videos**  
- Labels: continuous shooting-form scores (0–100)  
- Annotation: manually scored by the author based on coaching experience  

All videos originate from the same instructional environment and camera setup,
ensuring consistency across samples.

---

## Input Data Specification

- Input format: MP4 video files  
- Visual representation: RGB frames (as used by YOLOv8-Pose)  
- Camera setup:
  - single fixed camera  
  - sagittal plane (side view)  
- Subjects:
  - youth and amateur players  
  - real instructional class environment  

In each frame, **only the player performing the shot is used**
to construct a consistent pose time series.

---

## Pose Estimation with YOLOv8-Pose

Human pose keypoints are extracted using **YOLOv8-Pose**.

- Output: 2D keypoints per frame  
- Key joints used:
  - shoulder  
  - elbow  
  - wrist  
  - hip  

Frames without valid detections are skipped to ensure signal stability.

---

## Biomechanics-Informed Feature Design

From pose keypoints, the following **interpretable time-series features** are extracted:

- elbow joint angle trajectory  
- wrist vertical position normalized by torso length  
- reference baseline signal (for alignment)  

These features reflect biomechanically meaningful components of shooting mechanics,
including elbow path consistency, release height, and lower-body stability.

Although the number of input features is intentionally limited,
the model learns **temporal patterns and interactions** among them.

---

## Temporal Window Definition for Feature Extraction

Instead of learning raw temporal dynamics end-to-end,
this project uses a **fixed temporal window definition**.

- Start: deep motion initiation of the shooting action  
- End: just after ball release, before postural breakdown  

All videos are processed using the same temporal definition.
Extracted signals are:

- resampled to **80 time steps**
- z-score normalized per sequence  

This ensures temporal alignment while keeping the model architecture simple.

---

## Model: XGBoost Regression

### Why Regression Instead of Classification

Unlike Project 5, which framed the task as GOOD/BAD classification,
this project models shooting quality as a **continuous regression problem**.

This allows the system to:

- capture fine-grained differences in shooting form  
- provide more informative feedback  
- align more naturally with real coaching evaluation  

---

### Feature Vector Construction

Each shooting video is converted into a **240-dimensional feature vector**:

- elbow angle trajectory (80)  
- normalized wrist height (80)  
- reference baseline signal (80)  

The signals are concatenated after temporal resampling and normalization.

---

### XGBoost Architecture and Hyperparameters

```python
XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=42
)
```

## Key Design Choices

The XGBoost model is configured with the following design principles:

- **Shallow trees** to prevent memorization and overfitting  
- **Many estimators** to improve ensemble stability  
- **Low learning rate** for smooth and gradual optimization  
- **Subsampling** to reduce variance and improve generalization  

---

## Why XGBoost Works Well in This Project

XGBoost is well suited for this setting because:

- The dataset size is relatively small (**N = 107**)  
- Input features are already **strongly domain-informed**  
- Temporal information is **implicitly embedded** via fixed-length feature vectors  
- Tree-based models effectively handle **nonlinear feature interactions**  

As a result, the model generalizes more reliably than deep temporal models  
under limited data conditions.

---

## Training and Evaluation Protocol

- **5-fold cross-validation**  

### Metrics
- Mean Absolute Error (**MAE**)  
- **R² score**

A final model is trained on the full dataset  
and saved for downstream inference.

---

## Explainability with SHAP

To interpret model predictions, **SHAP (TreeExplainer)** is applied.

SHAP values are aggregated over temporal segments to estimate the  
relative contribution of the following components:

- **Elbow mechanics**  
- **Wrist control**  
- **Lower-body stability**

This enables transparent, **component-level analysis** of each prediction.

---

## AI Coach Feedback Design

Rather than generating generic natural-language explanations,  
the system produces **coach-style feedback**.

Feedback is derived from **relative SHAP contributions** and is phrased to reflect  
real coaching priorities, such as:

- elbow path consistency  
- wrist release control  
- lower-body balance  

This bridges the gap between model output  
and **actionable training guidance**.

---

## Experimental Results

### 5-Fold Cross-Validation (N = 107)

| Metric | Mean ± Std |
|------|-----------|
| MAE | 7.2 ± 0.4 points |
| R² | 0.59 ± 0.04 |

On average, the model predicts shooting-form scores within  
approximately **±7 points** of the ground-truth labels.

---

## Comparison with Project 5

### Project 5
- CNN + LSTM  vs XGBoost
- Explicit temporal modeling  
- Classification (GOOD / BAD)

### Project 6
- XGBoost regression  
- Temporal information embedded via feature design  
- Continuous scoring + explainability  

This comparison highlights how **model choice should align with data regime  
and task formulation**.

---

## Limitations

- Limited dataset size  
- Reliance on a single camera view  
- No subject-specific calibration  

Despite these constraints, the system demonstrates  
the feasibility of **explainable, pose-based shooting analysis**.

---

## Conclusion

This project demonstrates that,  
under limited data conditions,  
a carefully designed **feature-based machine learning approach**  
can outperform more complex temporal deep learning models  
in terms of **stability and interpretability**.

By integrating basketball domain knowledge,  
biomechanical reasoning, and explainable AI techniques,  
the system provides a **practical foundation**  
for AI-assisted shooting coaching.

---

## Technologies Used

### Programming & Libraries
- Python  
- PyTorch  
- OpenCV  
- scikit-learn  
- XGBoost  

### Models & Frameworks
- YOLOv8-Pose  
- 1D CNN (Project 5 comparison)  
- LSTM (Project 5 comparison)  

### Explainability & Deployment
- SHAP  
- Streamlit  

---

## Author

**Jungmyung Lee**  
Developed independently during the **LifeStep internship**  
Over **four years of basketball coaching experience**

