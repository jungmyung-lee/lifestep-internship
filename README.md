# Lifestep-Internship

## Overview 

This repository presents human motion analysis projects,
ranging from **3D biomechanical visualization** and **EMG signal processing**
to deep learning- and machine-learning–based **basketball shooting form classification** and **AI basketball coach systems.**

1️⃣ 3D Motion Visualization & Skeleton Reconstruction  
2️⃣ Automatic Gait Cycle Detection   
3️⃣ EMG preprocessing & Motion Synchronization  
4️⃣ Real-Time EMG-Based Prosthetic Hand Control  
5️⃣ Basketball Shooting Form Classification (GOOD / BAD)    
6️⃣ AI Basketball Shooting Coach


---
## Image & GIF

### 1.3D Motion Visualization & Skeleton Reconstruction  

## 3D Cutting Motion Visualization
![cuttingmat](https://github.com/user-attachments/assets/22c562bd-aa93-4b08-9142-d170baec23f2)

## 3D Squat Motion Visualization
![squatmat](https://github.com/user-attachments/assets/8eea3af3-1ae2-4eb2-b616-00ae1106e36c)

### 2. Automatic Gait Cycle Detection

<img width="408" height="308" alt="image" src="https://github.com/user-attachments/assets/d2fe5268-e0c3-48a4-9d00-997ba2707545" />
<img width="996" height="361" alt="image" src="https://github.com/user-attachments/assets/6b47925e-caa9-41be-ab00-43c0fe728bf4" />


### 3. EMG preprocessing & Motion Synchronization

<img width="1065" height="381" alt="image" src="https://github.com/user-attachments/assets/3f099fee-8dc6-4d48-8e01-8f858edbdfcb" />
<img width="1064" height="367" alt="image" src="https://github.com/user-attachments/assets/05563495-b6f7-4c03-95de-c943e4e78e69" />
<img width="1060" height="388" alt="image" src="https://github.com/user-attachments/assets/7f8903f2-2819-4e3e-9fe2-61d614daa5ae" />


## 4. Real-Time EMG-Based Prosthetic Hand Control (TD + CNN + TCN)

### Overview
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/dc879e80-de27-4032-b64c-bc1c4824df27" />



### Data Specification (Data sourced from LifeStep Company)

- **Measurement device (Input):**  
  Wearable surface EMG sensors (8-channel forearm EMG)

- **Training labels (Output):**  
  17 Hand joint angles obtained from synchronized motion-capture data  

### Input(Raw EMG 8 Channels of Muscles)  
  
Flexor Digitorum Superficialis (FDS) 
, Flexor Digitorum Profundus (FDP) 
, Flexor Carpi Radialis (FCR) 
, Flexor Carpi Ulnaris (FCU) 
, Extensor Digitorum (ED) 
, Extensor Carpi Radialis Longus (ECRL) 
, Extensor Carpi Ulnaris (ECU) 
, Pronator Teres (PT) 

![Flexor원본 복사본](https://github.com/user-attachments/assets/a51361d9-8fc3-41cb-8a91-59bcaa245e32)
<img width="620" height="440" alt="Extensor쪽-복사본-2" src="https://github.com/user-attachments/assets/d93d9a85-94ef-4cfc-a4dd-201aa7e709cd" />

*Image source: 

(Left) ScienceDirect – _Flexor Digitorum Superficialis Muscle* _ 
https://www.sciencedirect.com/topics/veterinary-science-and-veterinary-medicine/flexor-digitorum-superficialis-muscle

(Right) SlideServe – _*12 muscles 1. Anconeus 2. Brachioradialis (BR)*_  
https://www.slideserve.com/chet/12-muscles-1-anconeus-2-brachioradialis-br  


### Output(A single 17-dimensional vector representing continuous hand joint angles)

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/b24aeb65-1a98-42b3-aac2-5e4b5d48d79e" />


<div style="display: flex; gap: 16px; align-items: flex-start;">
  <img 
    src="https://github.com/user-attachments/assets/d713ca33-b9ff-4edd-a9d8-0212755719da" 
    alt="Hand anatomy model"
    width="400"
  />
  <img 
    src="https://github.com/user-attachments/assets/d5cab7fa-3413-4a60-a790-70c4a76d2cb8" 
    alt="Wrist anatomy diagram"
    width="550"
  />
</div>


(Left) Cayman Yoga Club – *3Gear Hand Model (Hand Anatomy & DOF)*  
https://www.caymanyogaclub.com/latest/doc/handModel.html  

(Right) Musculoskeletal Key – *Structure and Function of the Wrist*  
https://musculoskeletalkey.com/structure-and-function-of-the-wrist/




## 5,6. Basketball Shooting Form Classification (GOOD / BAD) & AI Basketball Shooting Coach
**Data sourced from the LJM Basketball Academy (founded and operated by the author)**

### YOLOv8 Pose Estimation
<p align="center">
  <img src="./yolov8.gif" width="300">
  <img src="./yolov8_2.gif" width="300">
  <img src="./yolov8_3.gif" width="300">
</p>


## Basketball Shooting Form Classification (GOOD / BAD) User Interface
![Basketball Shooting Form Classification UI](https://raw.githubusercontent.com/jungmyung-lee/lifestep-internship/main/13.gif)

## AI Basketball Shooting Coach Program User Interface
![AI Basketball Shooting Coach UI](https://raw.githubusercontent.com/jungmyung-lee/lifestep-internship/main/22.gif)

---

## Description


## 1. 3D Motion Visualization & Skeleton Reconstruction
This module focuses on 3D marker-based motion-capture visualization using Qualisys-style .mat datasets (e.g., squat.mat, cutting.mat).
The goal is to transform raw marker trajectories into interpretable **3D skeletal representations**, 
enabling visual inspection of human movement patterns before downstream biomechanical or machine-learning analysis.

🔍 **Detailed Information**  
👉 [Click here to view Project 1 detailed README](./project1/README.md)

---

## 2. Automatic Gait Cycle Detection

Using **OpenSim inverse kinematics output files (`.mot`)**, I implemented scripts to:

- Load and parse joint kinematic time-series data  
- Analyze knee joint angle time-series data for gait cycle detection  
- In the broader biomechanics pipeline, I also worked with additional  
  joint-angle and kinematic variables **(e.g., hip and ankle angles,  
  pelvis kinematics) for movement analysis**  
- Automatically detect and segment gait cycles using biomechanics-based signal processing
- Identify key movement phases from continuous joint-angle trajectories

- 🔍 **Detailed Information**  
👉 [Click here to view Project 2 detailed README](./project2/README.md)

---


## 3. EMG preprocessing & Motion Synchronization

This module performs full EMG preprocessing and synchronizes the processed signals with 3D marker-based motion data collected during squat and cutting tasks.

**Pre-processing Pipeline**

Five raw EMG channels are processed using:

-DC offset removal

-60 Hz notch filtering

-Band-pass filtering (physiological EMG range)

-Full-wave rectification

-RMS + envelope extraction

-These steps produce clean, analysis-ready activation signals.

### EMG–Motion Synchronization

Processed EMG envelopes are time-aligned with 3D marker trajectories (barbell markers + lower-body markers).

This makes it possible to compare muscle activation with movement phases.

Example observations:
Quadriceps activation peaks consistently at the bottom of each squat cycle.

From this synchronized analysis, users can examine:
-Muscle activation timing
-Left–right asymmetry or compensation patterns
-Movement efficiency

### Cutting Movement Analysis

Although the repository currently shows squat examples,
the same EMG–motion synchronization workflow was applied to cutting tasks to analyze activation strategies in more dynamic movements.

🔍 **Detailed Information**  
👉 [Click here to view Project 3 detailed README](./project3/README.md)


---





## 4. Real-Time EMG-Based Prosthetic Hand Control (TD + CNN + TCN)

I developed a lightweight real-time regression model using:

- Pre-processing(**DC offset removal,** **Band-pass filtering** (20–450Hz), **Notch filtering** (60 Hz), **MVC Normalization**)
- Time-Domain (TD) features (**MAV, WL, ZC, SSC, RMS**)
- ****1D** CNN** layers for spatial pattern extraction
- **TCN** layers with dilation for temporal modeling  

The model processes **EMG signals as input** and performs inference every **0.01 seconds (100 Hz)**, enabling real-time operation.  
The final output consists of **17 joint angles**, representing multi-DOF finger and wrist movements.

This repository includes the full model architecture and training code used to build the **prosthetic control system**.

🔍 **Detailed Information**  
👉 [Click here to view Project 4 detailed README](./project4/README.md)


---

## 5. Basketball Shooting Form Classification (GOOD / BAD) (CNN + LSTM vs XGBoost)

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
when training temporal deep models with **small labeled datasets**.

🔍 **Detailed Information**  
👉 [Click here to view Project 5 detailed README](./project5/README.md)

---

## 6. AI Basketball Shooting Coach (XGBoost Regression + SHAP)

Building upon the pose-based shooting analysis developed in Project 5,
I implemented an end-to-end **AI basketball shooting coach** that predicts
a continuous shooting-form score and provides **explainable, body-part-level feedback**.

From each shooting video, pose-based features were extracted using **YOLOv8-Pose**
and used to train an **XGBoost regression model** that predicts a numeric
shooting-form score on a 0–100 scale.
To improve interpretability, **SHAP** was applied to estimate the relative
importance of different body components (elbow, wrist, lower body) for each prediction.

A lightweight **Streamlit-based web interface** was implemented to allow users
to upload a shooting video and receive a score along with interpretable,
coach-style feedback.

### Results (5-Fold Cross-Validation, N = 107)

| Metric | Value (mean ± std) |
|------|---------------------|
| MAE  | 7.2 ± 0.4 points    |
| R²   | 0.59 ± 0.04         |

On average, the model predicts shooting-form scores within approximately
±7 points of the ground-truth labels and explains around 60% of the variance
in shooting-form quality, demonstrating the feasibility of pose-based,
explainable scoring under a small-data setting.

🔍 **Detailed Information**  
👉 [Click here to view Project 6 detailed README](./project6/README.md)







