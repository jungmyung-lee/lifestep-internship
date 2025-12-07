# lifestep-internship

This repository contains Python implementations from my biomechanics internship at **LifeStep**, focusing on three core areas:  
(1) joint-angle extraction & 3D visualization,  
(2) EMG preprocessing,  
(3) AI model development for prosthetic control.

---

1. 3D visualization
### 3D Cutting Motion Visualization
![Cutting Motion](./cutting.gif)

### 3D Squat Motion Visualization
![Squat Motion](./squat.gif)


2.EMG preprocessing

<img width="1333" height="553" alt="image" src="https://github.com/user-attachments/assets/11eca52d-4899-48cb-92c9-5cdaf817dd56" />
<img width="1311" height="502" alt="image" src="https://github.com/user-attachments/assets/29089095-18b1-455c-9403-2072dba75b19" />
<img width="1331" height="456" alt="image" src="https://github.com/user-attachments/assets/b79e7c8a-4ef5-4955-8591-e73f3559c136" />
<img width="1335" height="477" alt="image" src="https://github.com/user-attachments/assets/53bdab7d-8e23-41cc-a973-8ca8cc5f83cf" />



## 1. Joint Angle Extraction & 3D Motion Visualization

Using datasets such as **squat.mat** and **cutting.mat**, I implemented scripts to:

- Load and parse motion-capture data  
- Extract joint variables including:  
  - Knee angle  
  - Hip and ankle angles  
  - Pelvis kinematics  
- Identify key movement phases (squat cycles, cutting phases)

I also reconstructed 3D marker trajectories and generated **3D motion GIFs**, enabling intuitive visualization of the movements.

---

## 2. EMG Signal Pre-processing

I processed **five channels of raw EMG** using a complete preprocessing pipeline:

- Band-pass filtering  
- Rectification  
- RMS and envelope extraction  
- Visualization of raw vs. processed EMG signals  

These processed signals provide clean, model-ready features.

---

## 3. TD + CNN + TCN Real-Time Model (17-Angle Output)

I developed a lightweight real-time regression model using:

- Time-Domain (TD) features  
- 1D CNN layers for spatial pattern extraction  
- TCN layers with dilation for temporal modeling  

The model processes **EMG signals as input** and performs inference every **0.01 seconds (100 Hz)**, enabling real-time operation.  
The final output consists of **17 joint angles**, representing multi-DOF finger and wrist movements.

This repository includes the full model architecture and training code used to build the prosthetic control system.


This repository demonstrates an end-to-end workflow from biomechanics data processing to EMG feature extraction and real-time machine-learning modeling.
