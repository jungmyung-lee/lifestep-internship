# 🧠 lifestep-internship

This repository contains Python implementations developed during my biomechanics internship
at **LifeStep**, focusing on **human motion analysis and biosignal-driven modeling** across
multiple application domains.

1️⃣ 3D Visualization  
2️⃣ Joint-angle extraction (including other biomechanical variables)  
3️⃣ EMG preprocessing & Motion Synchronization  
4️⃣ AI model development for prosthetic control
5️⃣ Pose-Based Basketball Shooting Form Classification (GOOD / BAD)
6️⃣ Explainable AI Basketball Shooting Coach


---

### 1.🎥  3D visualization (GIF) 

## 🔹3D Cutting Motion Visualization
![Cutting Motion](./cutting.gif)

## 🔹3D Squat Motion Visualization
![Squat Motion](./squat.gif)

### 2. joing-angle extraction (inculding other biomechanical variables)

<img width="408" height="308" alt="image" src="https://github.com/user-attachments/assets/d2fe5268-e0c3-48a4-9d00-997ba2707545" />
<img width="996" height="361" alt="image" src="https://github.com/user-attachments/assets/6b47925e-caa9-41be-ab00-43c0fe728bf4" />



### 3.EMG preprocessing

<img width="1333" height="553" alt="image" src="https://github.com/user-attachments/assets/11eca52d-4899-48cb-92c9-5cdaf817dd56" />
<img width="1311" height="502" alt="image" src="https://github.com/user-attachments/assets/29089095-18b1-455c-9403-2072dba75b19" />
<img width="1315" height="484" alt="image" src="https://github.com/user-attachments/assets/15c20aab-99d1-4b9c-9c16-c84bafeafc66" />
<img width="1298" height="495" alt="image" src="https://github.com/user-attachments/assets/f7e61767-b803-46bb-8333-bee8784bcae4" />


## 2. Joint Angle Extraction & 3D Motion Visualization

Using datasets such as **squat.mat** and **cutting.mat**, I implemented scripts to:

- Load and parse motion-capture data  
- Extract joint variables including:  
  - Knee, hip, and ankle angles  
  - Pelvis kinematics  
  - Additional biomechanical variables available in the .mat datasets (e.g., GRF, marker trajectories, etc.) 
- Identify key movement phases (squat cycles, cutting phases)

I also reconstructed 3D marker trajectories and generated **3D motion GIFs**, enabling intuitive visualization of the movements.

---


## ✨ 3. EMG Signal Pre-processing & Motion Synchronization

This module performs full EMG preprocessing and synchronizes the processed signals with 3D marker-based motion data collected during squat and cutting tasks.

⚙️ Pre-processing Pipeline

Five raw EMG channels are processed using:

-DC offset removal

-60 Hz notch filtering

-Band-pass filtering (physiological EMG range)

-Full-wave rectification

-RMS + envelope extraction

-These steps produce clean, analysis-ready activation signals.

### 🎯 EMG–Motion Synchronization

Processed EMG envelopes are time-aligned with 3D marker trajectories (barbell markers + lower-body markers).

This makes it possible to compare muscle activation with movement phases.

Example observations:
Quadriceps activation peaks consistently at the bottom of each squat cycle.

From this synchronized analysis, users can examine:
-Muscle activation timing
-Left–right asymmetry or compensation patterns
-Movement efficiency

### 🏃‍♂️ Cutting Movement Analysis

Although the repository currently shows squat examples,
the same EMG–motion synchronization workflow was applied to cutting tasks to analyze activation strategies in more dynamic movements.

## 4. TD + CNN + TCN Real-Time Model (prosthetic control system)

I developed a lightweight real-time regression model using:

- Time-Domain (TD) features  
- 1D CNN layers for spatial pattern extraction  
- TCN layers with dilation for temporal modeling  

The model processes **EMG signals as input** and performs inference every **0.01 seconds (100 Hz)**, enabling real-time operation.  
The final output consists of **17 joint angles**, representing multi-DOF finger and wrist movements.

This repository includes the full model architecture and training code used to build the **prosthetic control system**.

## 5. Pose-Based Basketball Shooting Form Classification (GOOD / BAD)

I independently built **a basketball shooting-form classification** pipeline using **YOLOv8-Pose** during my internship at LifeStep.
From each shooting video, I extracted 2D keypoints and derived interpretable time-series features
(e.g., elbow angle trajectory and normalized wrist height). Each sequence was resampled to a fixed
length and normalized.

To evaluate modeling choices under a small annotated dataset setting **(N = 107)**,
I compared two approaches:

- **CNN + LSTM**: temporal deep model to learn motion dynamics directly from the resampled sequences  
- **XGBoost**: tree-based classifier trained on flattened pose-derived feature vectors  

### Results (Stratified 5-Fold Cross-Validation, N = 107)

| Model      | Accuracy (mean ± std) | F1-score (mean ± std) |
|------------|------------------------|------------------------|
| CNN + LSTM | 0.66 ± 0.07             | 0.64 ± 0.08            |
| XGBoost    | **0.74 ± 0.04**         | **0.73 ± 0.05**        |​

In this limited-data regime, **XGBoost showed more stable generalization across folds**,
while the CNN+LSTM model exhibited higher variance, suggesting sensitivity to overfitting
when training temporal deep models with small labeled datasets.

## 6. Explainable AI Basketball Shooting Coach (End-to-End System)

Building upon the pose-based shooting analysis developed in Project 6,
I implemented an end-to-end **AI basketball shooting coach** that predicts
a numeric shooting-form score and provides **explainable, body-part-level feedback**.

### System Overview

The system consists of four main components:

- **Feature extraction**  
  Shooting videos are processed using **YOLOv8-Pose** to extract 2D body keypoints.
  Interpretable biomechanical features—such as elbow joint angles and normalized
  wrist height—are computed, resampled to a fixed temporal length, and z-score normalized.

- **Regression model**  
  An **XGBoost regressor** is trained to predict a continuous shooting-form score
  on a 0–100 scale using pose-derived feature vectors.
  Model performance is evaluated using **5-fold cross-validation**, and a final
  model is trained on the full dataset and saved for inference.

- **Explainable AI (SHAP)**  
  SHAP TreeExplainer is applied to estimate the relative contribution of different
  body components (elbow, wrist, lower body) to the predicted score.
  These contributions are aggregated across time to enable stable,
  component-level interpretation.

- **Interactive web interface**  
  A **Streamlit-based web application** allows users to upload a shooting video,
  run the full analysis pipeline, and receive:
  - An overall shooting-form score  
  - A joint-level contribution breakdown  
  - Natural-language feedback generated from SHAP-based importance ratios  

### Results (5-Fold Cross-Validation, N ≈ 107)

The regression model was evaluated using 5-fold cross-validation.
Performance was measured using **Mean Absolute Error (MAE)** and **R² score**.

| Metric | Value (mean ± std) |
|------|---------------------|
| MAE  | 7.2 ± 0.4 points    |
| R²   | 0.59 ± 0.04         |

On average, the model predicts shooting-form scores within approximately
±7 points of the ground-truth labels and explains around 60% of the variance
in shooting-form quality.

### Key Characteristics

- End-to-end pipeline from **raw video → score prediction → explainable feedback**
- Explicit separation between **prediction** and **interpretation**
- Designed for **small annotated datasets**, emphasizing robustness and interpretability
- Demonstrates how biomechanics-inspired features can support
  **explainable AI systems** for sports performance analysis






