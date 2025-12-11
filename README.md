# 🧠 lifestep-internship

This repository contains Python implementations from my biomechanics internship at **LifeStep**, focusing on three core areas:  
1️⃣ joint-angle extraction (inculding other biomechanical variables) & 3D visualization,  
2️⃣ EMG preprocessing,  
3️⃣ AI model development for prosthetic control.

---

### 1.🎥  3D visualization (GIF)
## 🔹3D Cutting Motion Visualization
![Cutting Motion](./cutting.gif)

## 🔹3D Squat Motion Visualization
![Squat Motion](./squat.gif)


### 2.EMG preprocessing (IMG)

<img width="1333" height="553" alt="image" src="https://github.com/user-attachments/assets/11eca52d-4899-48cb-92c9-5cdaf817dd56" />
<img width="1311" height="502" alt="image" src="https://github.com/user-attachments/assets/29089095-18b1-455c-9403-2072dba75b19" />
<img width="1315" height="484" alt="image" src="https://github.com/user-attachments/assets/15c20aab-99d1-4b9c-9c16-c84bafeafc66" />
<img width="1298" height="495" alt="image" src="https://github.com/user-attachments/assets/f7e61767-b803-46bb-8333-bee8784bcae4" />


## 1. Joint Angle Extraction & 3D Motion Visualization

Using datasets such as **squat.mat** and **cutting.mat**, I implemented scripts to:

- Load and parse motion-capture data  
- Extract joint variables including:  
  - Knee, hip, and ankle angles  
  - Pelvis kinematics  
  - Additional biomechanical variables available in the .mat datasets (e.g., GRF, marker trajectories, etc.) 
- Identify key movement phases (squat cycles, cutting phases)

I also reconstructed 3D marker trajectories and generated **3D motion GIFs**, enabling intuitive visualization of the movements.

---


## ✨ 2. EMG Signal Pre-processing & Motion Synchronization

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

## 3. TD + CNN + TCN Real-Time Model (17-Angle Output)

I developed a lightweight real-time regression model using:

- Time-Domain (TD) features  
- 1D CNN layers for spatial pattern extraction  
- TCN layers with dilation for temporal modeling  

The model processes **EMG signals as input** and performs inference every **0.01 seconds (100 Hz)**, enabling real-time operation.  
The final output consists of **17 joint angles**, representing multi-DOF finger and wrist movements.

This repository includes the full model architecture and training code used to build the prosthetic control system.


This repository demonstrates an end-to-end workflow from biomechanics data processing to EMG feature extraction and real-time machine-learning modeling.
