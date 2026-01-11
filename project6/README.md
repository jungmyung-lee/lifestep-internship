# AI Basketball Shooting Coach  
### Pose-Based Shooting Form Scoring with XGBoost Regression + SHAP

![Cutting Motion](https://raw.githubusercontent.com/jungmyung-lee/lifestep-internship/main/22.gif)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Dataset and Data Curation](#dataset-and-data-curation)
- [Pose Estimation with YOLOv8-Pose](#pose-estimation-with-yolov8-pose)
- [Biomechanics-Informed Feature Design](#biomechanics-informed-feature-design)
- [Temporal Window Definition for Feature Extraction](#temporal-window-definition-for-feature-extraction)

- [Model: XGBoost Regression](#model-xgboost-regression)
  - [Why Regression Instead of Classification](#why-regression-instead-of-classification)
  - [Feature Vector Construction](#feature-vector-construction)
  - [XGBoost Architecture and Hyperparameters](#xgboost-architecture-and-hyperparameters)
  - [Key Design Choices (Detailed Rationale)](#key-design-choices-detailed-rationale)
  - [Why XGBoost Works Well in This Project](#why-xgboost-works-well-in-this-project)

- [Training and Evaluation Protocol](#training-and-evaluation-protocol)
- [Explainability with SHAP](#explainability-with-shap)
- [AI Coach Feedback Design](#ai-coach-feedback-design)
- [Experimental Results](#experimental-results)
- [Comparison with Project 5](#comparison-with-project-5)
- [Limitations and Reflections](#limitations-and-reflections)
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

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/2bf9f4bf-b358-4799-bdc2-7dac22653a8c" />

## Dataset and Data Curation

- Data source: real basketball class session videos  
- Total samples: **107 shooting videos**  
- Labels: **continuous shooting-form scores (0–100)**  
- Annotation: manually scored by the author based on basketball coaching experience  

All labels were **manually assigned by the author**,  
who has **over four years of basketball coaching experience**.


### Data Characteristics and Recording Setup

The dataset was collected in a **controlled instructional environment**,  
rather than an unconstrained real-game setting.

Key characteristics include:

- Input format: **MP4 video files**
- Visual representation: **RGB frames** (as used by YOLOv8-Pose)
- All videos were recorded using a **single fixed camera**
- The camera view was restricted to a **sagittal (side) view**
- No multi-view or multi-angle recordings were used
- Recording conditions (court, lighting, background) were largely consistent

### Subjects and Sample Definition

- Participants consist of **youth and amateur basketball players**
- Each shooting video corresponds to **one complete shooting motion**
  performed by a **single individual**
- No repeated shots from the same individual are included,
  minimizing subject-specific bias and intra-subject correlation

In each frame, **only the player performing the shot is used**
to construct a consistent pose-based time series,
ensuring temporal and spatial coherence across the sequence.

As a result, the dataset represents a **low-variance, domain-specific motion setting**.  
The reported model performance should therefore be interpreted  
in the context of these controlled data conditions,
rather than as a measure of unconstrained real-world generalization.


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
- normalized hip vertical displacement (proxy for lower-body stability)  


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

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/52a874dd-84a6-4884-8df9-866cacc29e0d" />

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
- normalized hip vertical displacement (80)   

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

## Key Design Choices (Detailed Rationale)

### **max_depth = 4** (Shallow Trees)

The dataset size is relatively small (**N = 107**), which makes deep trees prone to memorizing noise and sample-specific patterns.

The input features consist of **time-resampled biomechanical signals**
(elbow angle, wrist height, hip vertical displacement), flattened into fixed-length vectors.
Excessively deep trees could overfit to highly localized conditions,
such as values at specific time indices.

A depth of **4** provides a practical balance:

- Deep enough to capture meaningful **nonlinear interactions**
  (e.g., combinations of elbow and wrist patterns across motion phases)
- Shallow enough to limit unnecessary branching that increases overfitting risk


---

### **n_estimators = 400** (Many Boosting Rounds)

Instead of relying on a few complex trees,
the model uses **many shallow trees** that are incrementally combined,
following the core boosting strategy.

With a small dataset, large updates can destabilize training.
Using more estimators allows the model to **refine predictions gradually**,
which is particularly important for regression on a **continuous 0–100 score scale**.

This design helps produce **smoother and more stable predictions**
across cross-validation folds.


---

### **learning_rate = 0.03** (Low Learning Rate)

A low learning rate ensures that each tree makes only a **small correction**
to the current prediction.

In small-data settings, higher learning rates (e.g., ≥ 0.1)
can cause aggressive updates that overfit individual samples
and lead to **high variance across folds**.

The combination of a **low learning rate** with **many estimators**
is a standard stabilization strategy for reliable generalization.


---

### **subsample = 0.9** (Row Sampling)

Each tree is trained using a **random subset (90%)** of the training samples.

This introduces controlled randomness across trees,
reducing correlation within the ensemble and mitigating overfitting.
Given the limited dataset size, a very low subsample ratio
could result in excessive information loss.

A value of **0.9** provides regularization benefits
while preserving most of the available data.


---

### **colsample_bytree = 0.9** (Feature Sampling)

Each tree randomly selects **90% of the available features**
from the **240-dimensional input vector**.

Because the features are temporally ordered,
adjacent features tend to be **highly correlated**.
Feature sampling discourages over-reliance on specific time indices
and encourages learning **robust feature combinations**
across different temporal regions.


---

### **objective = "reg:squarederror"** (Regression Loss)

The task is formulated as a **continuous regression problem**
predicting shooting-form scores on a **0–100 scale**.

The squared error objective penalizes **large prediction errors more strongly**,
encouraging the model to reduce extreme mispredictions
and produce **stable score estimates**.

<img width="186" height="41" alt="image" src="https://github.com/user-attachments/assets/76898b99-eefa-4a80-a8f6-e8960fcb3324" />



---

### **random_state = 42** (Reproducibility)

With a small dataset, training results can vary
depending on data splits and sampling.

Fixing the random seed ensures **experimental reproducibility**
and consistency between reported results
(e.g., **MAE and R²**) and future model runs.


---

### **Relation to XGBoost in Project 5 (Classification Setting)**

Compared to the XGBoost classifier used in **Project 5**,
this project focuses on **continuous score regression**,
which requires capturing more fine-grained variations in shooting form.

For this reason, the tree depth is **slightly increased**
while maintaining strong regularization
through a **low learning rate** and **subsampling**.

This design enables **stable learning under the same small-data regime**
(**N = 107**) without sacrificing generalization.


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

## Limitations and Reflections

### Limited Dataset Size and Controlled Data Conditions

The dataset used in this project is relatively small (**N = 107**) and was collected
in a **controlled instructional environment** with a single fixed camera
and a sagittal (side) view.

While this setup ensures consistency and low noise in pose extraction,
it limits viewpoint diversity and environmental variability.
As a result, the reported regression performance
(e.g., MAE and R²) should not be interpreted as
direct indicators of unconstrained real-world generalization.

Under broader and more diverse recording conditions,
model performance is expected to decrease initially,
but would provide a more realistic estimate of robustness.


---

### Subjectivity in Score Annotation

The shooting-form scores in this project were
**manually assigned by the author**, who has
**over four years of basketball coaching experience**.

Basketball shooting evaluation is inherently subjective
and may vary across coaches depending on
instructional philosophy, evaluation criteria,
and emphasis on specific biomechanical components.

Consequently, the model learns to approximate
a **specific coaching perspective**
rather than an objective or universal ground truth.
Different coaches may assign different scores
to the same shooting motion.

For this reason, the current system should be viewed as
an **early-stage decision-support tool**
rather than a definitive or standardized evaluator.


---

### Reflection and Future Directions

Despite these limitations, this project demonstrates that
coaching knowledge can be systematically translated into
a **quantitative, interpretable regression framework**
using pose-based features and explainable machine learning.

Future improvements may include:

- incorporating **multi-coach score annotations**
- establishing more standardized scoring criteria
- expanding the dataset with more diverse viewpoints and players

These extensions would help improve both
the objectivity and generalization of the scoring system.


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

