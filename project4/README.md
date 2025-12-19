# Project4  
# **Real-Time EMG Regression Pipeline**(TD + CNN + TCN)  

## Table of Contents

- [Project Overview](#project-overview)
- [Readme](#readme)
- [Data Specification](#data-specification-data-sourced-from-lifestep-company)
- [Research Basis](#research-basis)
- [Model Purpose](#model-purpose)
- [Overall System Pipeline](#overall-system-pipeline)

- [EMG Input & Pre-processing](#emg-input--pre-processing)
- [MVC-Based Robust Normalization](#mvc-based-robust-normalization)
- [Windowing Strategy (150 ms / 10 ms Hop)](#windowing-strategy-150-ms--10-ms-hop)
- [Time-Domain (TD) Feature Extraction](#time-domain-td-feature-extraction)

- [CNN Encoder](#cnn-encoder)
  - [Why 2 Layers](#why-exactly-2-cnn-layers)
  - [Why 16 → 32 Filters](#why-use-16--32-filters)
  - [Why Kernel Size = 3](#why-kernel-size--3)
  - [Why ReLU Activation](#why-relu-activation)
  - [Why Compact Feature Representation](#why-output-a-compact-feature-representation-32–48-dimensions)

- [Temporal Convolutional Network (TCN)](#temporal-convolutional-network-tcn)
  - [Why TCN over LSTM](#why-use-a-tcn-instead-of-an-lstm)
  - [Why 1–2 TCN Layers](#why-only-1–2-tcn-layers)
  - [Why Hidden Channels = 32](#why-hidden-channels--32)
  - [Why Dilations = 1, 2](#why-dilations--1-and-2-only)
  - [Why Kernel Size = 3 in the TCN](#why-kernel-size--3-in-the-tcn)

- [Loss Function](#loss-function)
- [Final Remark](#final-remark)
- [References](#references)



<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/82316845-b6c2-4d9f-ae7a-f704935a32bf" />


## Project Overview

This repository documents **Project4**, a real-time EMG regression system developed during the **LifeStep Internship**.  
The goal of this project is to design a **lightweight, low-latency, and robust pipeline** that converts **multi-channel surface EMG signals** into **continuous joint-angle estimates**, suitable for **embedded prosthetic hand control**.

Unlike purely offline or high-compute research models, this project explicitly targets **real-time wearable and embedded environments**, where **latency, computational cost, robustness to noise, and generalization across users** are critical constraints.

---

### Readme

**Project4 Presentation (PDF)**  
->[Open PDF](Project4_presentation.pdf)

---


### Data Specification (Data sourced from LifeStep Company)

- **Measurement device (Input):**  
  Wearable surface EMG sensors (8-channel forearm EMG)

- **Training labels (Output):**  
  17 Hand joint angles obtained from synchronized motion-capture data  

---

### Research Basis

This project is informed by five key research papers on EMG-based pattern recognition and prosthetic control.  
Among them, three papers played a central role:

1. **Real-Time EMG Based Pattern Recognition Control for Hand Prostheses (2019)**  
   – guidance on real-time processing and system design.

2. **A Review of Myoelectric Control for Prosthetic Hand Manipulation (2023)**  
   – insights into modern myoelectric control strategies.

3. **A New Strategy for Multifunction Myoelectric Control (1993)**  
   – foundation for selecting the TD feature set (MAV, RMS, WL, ZC, SSC).

Two additional papers were used for supplementary background.  
Full references are listed at the bottom of this README (APA style).

---

## Model Purpose

The primary objective is to map **raw surface EMG signals** to **continuous joint-angle outputs** with **minimal delay**.  
The predicted angles are intended to be **directly usable in prosthetic control loops**, where **unstable or delayed outputs significantly degrade usability**.

---

## Overall System Pipeline

The complete pipeline is structured as:
Raw EMG
→ Pre-processing
→ MVC-based Normalization
→ Windowing (150 ms / 10 ms hop)
→ Segmentation (3 segments)
→ TD Feature Extraction
→ CNN Encoder
→ TCN Temporal Model
→ Fully Connected Regression
→ Continuous Joint-Angle Output


Each stage is deliberately designed to **preserve physiological meaning while minimizing computational overhead**.

---

## EMG Input & Pre-processing

- **Sampling rate:** 1000 Hz  
- **Number of channels:** 8  

### Filtering steps

- **Band-pass filter: 20–450 Hz**
  - Removes low-frequency motion artifacts and baseline drift
  - Suppresses high-frequency noise

- **Notch filter: 60 Hz**
  - Eliminates power-line interference

- **DC offset removal**

These steps isolate the **physiologically relevant EMG frequency band** and stabilize the signal before feature extraction.

---

## MVC-Based Robust Normalization

Surface EMG amplitude is highly sensitive to:

- subject differences  
- electrode placement  
- session variability  
- skin impedance  

To mitigate this, **MVC-based normalization** is applied.

### Procedure

1. Record a dedicated **100% MVC trial**
2. Compute a **per-channel reference gain (`ref_gain`)** using **RMS or MAV**
3. Normalize all EMG samples as:  
x_norm(t) = x(t) / ref_gain


### Why MVC normalization matters

- Aligns signal scale across users and sessions  
- Prevents the model from learning amplitude bias  
- Improves robustness and generalization in real-world deployment  

---

## Windowing Strategy (150 ms / 10 ms Hop)

- **Window length:** 150 ms (150 samples)  
- **Hop size:** 10 ms (100 Hz control loop)  

### Interpretation

- The first prediction is available after **150 ms**
- Subsequently, new predictions are produced **every 10 ms**

This design balances **muscle activation pattern capture** and **real-time responsiveness**, making it suitable for continuous prosthetic control.

---

## Time-Domain (TD) Feature Extraction

Each window is divided into **three 50 ms segments**.  
For each segment, five **Hudgins Time-Domain (TD) features** are extracted:

- **MAV** – amplitude  
- **RMS** – signal energy  
- **WL** – waveform complexity  
- **ZC** – frequency-related information  
- **SSC** – shape change information  

### Why this exact TD feature set?

- All features have **extremely low computational cost**
- Historically validated in **prosthetic EMG literature** (Hudgins et al., 1993)
- Well-balanced coverage of **amplitude, frequency, and waveform shape**
- Avoids heavy features (FFT, wavelets, entropy) that increase latency
- Minimizes downstream **CNN / TCN computational load**

### TD Final Feature Shape Before Entering the 1D CNN

- **3 segments × 5 features = 15 features per channel**
- CNN input tensor shape:
[Batch, Channels = 8, Time = 15]


## CNN Encoder  
### Why 2 Layers, 16 → 32 Filters, Kernel Size = 3, ReLU

A **two-layer 1D Convolutional Neural Network (CNN)** is used as a **feature encoder** that transforms TD-based EMG representations into a compact and informative latent feature space before temporal modeling.

### Why use a CNN for EMG signals?

Although CNNs are commonly associated with image processing, a **1D CNN operating along the temporal axis** is particularly well suited for EMG signals because:

- EMG contains **short-term local temporal patterns**, such as:
  - Motor Unit Action Potential (MUAP) shapes
  - brief muscle activation bursts
  - abrupt changes in activation level
- These waveform-level patterns are **difficult to fully encode using hand-crafted TD features alone**
- A CNN can automatically learn **local temporal structures** that remain relatively consistent across:
  - subjects
  - electrode placements
  - minor signal distortions

In this pipeline, the CNN acts as a **feature encoder**, compressing refined EMG + TD information into a **compact representation (approximately 32–48 dimensions)** that preserves muscle synergy information while suppressing noise.

---

### Why exactly 2 CNN layers?

#### Why 1 layer is insufficient
- A single CNN layer behaves similarly to a flexible FIR filter bank
- It can detect very local patterns (e.g., spikes or short bursts) but struggles to reliably separate:
  - true muscle activation
  - electrode motion artifacts
  - residual noise
- Because EMG signals are inherently noisy and heterogeneous, **one convolutional layer is not sufficient to organize the structure**

#### Why more than 2 layers are excessive
- EMG datasets are typically limited in:
  - recording duration
  - number of subjects
- Deeper CNNs lead to:
  - rapid parameter growth
  - increased overfitting risk
  - higher inference latency
  - increased power consumption
- For **real-time prosthetic control**, deep CNN architectures are unnecessary and inefficient

#### Why 2 layers provide the best balance
- **First layer:** extracts low-level waveform features  
  (e.g., edges, spikes, MUAP-like patterns)
- **Second layer:** recombines those features to distinguish:
  - muscle activation signatures
  - structured noise patterns
- This depth provides sufficient representational power while respecting real-time and embedded constraints

---

### Why use 16 → 32 filters?

#### First CNN layer (16 filters)
- Learns basic muscle activation patterns such as:
  - frequency-emphasized waveforms
  - sharp vs. smooth MUAP shapes
  - patterns repeated across channels
- With 8 EMG channels, **16 filters (~2× the channel count)** provide adequate diversity without overfitting noise

#### Second CNN layer (32 filters)
- Learns **combinational patterns**, including:
  - muscle synergies
  - task-specific co-activation patterns
- Increasing the filter count at this stage allows flexible recombination of first-layer features

#### Why not larger filter sizes (64, 128)?
- Larger filter counts cause:
  - parameter explosion
  - increased overfitting
  - higher computational cost
  - increased latency and power consumption
- **16 → 32** represents a controlled and efficient expansion of representational capacity suitable for EMG and embedded deployment

---

### Why kernel size = 3?

- A kernel size of **3 samples** captures:
  - the current time point
  - immediate past and future context
- This is sufficient to represent:
  - short EMG bursts
  - MUAP peaks
  - rapid activation transitions

Comparison with other kernel sizes:
- **k = 1:** ignores temporal structure and behaves like point-wise projection
- **k = 5 or 7:** increases parameters and blurs fine-grained waveform details

In this architecture:
- The **CNN focuses on short-range local patterns**
- Longer temporal dependencies are handled by the TCN  
→ kernel size **3** is the most natural and efficient choice

---

### Why ReLU activation?

- **Extremely lightweight computation**
  - no exponentials or divisions
  - well suited for ARM processors and microcontrollers
- **Well matched to EMG characteristics**
  - rectified EMG and TD features are largely non-negative
  - ReLU naturally suppresses small negative noise
- **Stable training behavior**
  - avoids vanishing gradients common in sigmoid or tanh

Overall, ReLU is a practical and robust activation function for **noisy EMG signals in real-time embedded systems**.

---

### Why output a compact feature representation (32–48 dimensions)?

- **Too large (e.g., 128–256 dimensions):**
  - overwhelms the TCN
  - increases parameters and overfitting
  - degrades real-time performance
- **Too small (e.g., 8–16 dimensions):**
  - discards important muscle synergy information

A **32–48 dimensional bottleneck** preserves essential activation structure while providing an efficient and stable input for temporal modeling.

---

## Temporal Convolutional Network (TCN)  
### Why 1–2 Layers, Hidden Channels = 32, Dilations = 1, 2, Kernel Size = 3

The **Temporal Convolutional Network (TCN)** models temporal dependencies across consecutive EMG windows after spatial encoding by the CNN.

---

### Why use a TCN instead of an LSTM?

- Comparable temporal modeling capability to LSTM
- Fully parallel computation (no sequential dependency)
- Lower inference latency
- Reduced power consumption

These properties make TCNs more suitable for **real-time prosthetic control** than recurrent architectures.

---

### Why only 1–2 TCN layers?

- EMG-based control primarily depends on **short temporal context (approximately 100–200 ms)**
- Deeper TCNs unnecessarily increase the receptive field
- Excessive depth leads to:
  - overfitting
  - degraded real-time performance
  - increased latency

Thus, **1–2 TCN layers** provide sufficient temporal modeling capacity without violating real-time constraints.

---

### Why hidden channels = 32?

- Hidden channels represent the **temporal modeling capacity** of the TCN
- EMG temporal patterns are inherently compact
- The CNN already outputs a **32–48 dimensional compressed representation**
- Larger values (64, 128) cause:
  - parameter explosion
  - overfitting
  - increased latency and power consumption

A hidden size of **32** offers the best balance between expressiveness and efficiency.

---

### Why dilations = 1 and 2 only?

- EMG temporal structure is **short-range**
- Large dilations skip meaningful MUAP-level patterns
- High dilation values increase sensitivity to:
  - noise
  - electrode placement shifts

Dilations of **1 and 2** are sufficient to cover the required temporal context for EMG-based control.

---

### Why kernel size = 3 in the TCN?

- Captures basic temporal transitions:
  - spikes
  - activation rise and fall
- Larger kernels increase computation without clear benefit
- Longer temporal dependencies are already handled through dilation

---

## Loss Function  
### Why Mean Squared Error (MSE)?

The model is trained using **Mean Squared Error (MSE) loss**, defined as:

\[
\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
\]

### Why MSE for EMG-based joint-angle regression?

- The task is a **continuous regression problem**, not classification
- Joint angles are real-valued signals where:
  - large prediction errors are particularly harmful
  - sudden deviations can destabilize prosthetic control
- MSE strongly penalizes large errors through squaring
- This encourages:
  - smooth predictions
  - stable control behavior

### Why not other loss functions?

- **MAE:** treats all errors equally and provides weaker gradients for fine adjustment
- **Huber loss:** introduces additional hyperparameters without clear benefit in this setting
- **Correlation-based losses:** do not directly minimize absolute angle error required for control

MSE provides a **simple, stable, and physically meaningful objective** for real-time prosthetic joint-angle regression.

---

## Final Remark

This README intentionally documents **not only what components were used, but why each design choice was made**.  
All architectural decisions and hyperparameters are selected based on **physiological EMG properties, real-time constraints, and embedded deployment considerations**.






### References

Al-Timemy, A. H., Bugmann, G., Escudero, J., & Outram, N. (2013). Classification of Finger Movements for the Dexterous Hand Prosthesis Control With Surface Electromyography. IEEE Journal of Biomedical and Health Informatics, 17(3), 608–618. https://doi.org/10.1109/JBHI.2013.2249590


Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. https://doi.org/10.48550/arxiv.1803.01271

Chen, Z., Min, H., Wang, D., Xia, Z., Sun, F., & Fang, B. (2023). A Review of Myoelectric Control for Prosthetic Hand Manipulation. Biomimetics (Basel, Switzerland), 8(3), 328. https://doi.org/10.3390/biomimetics8030328

Hudgins, B., Parker, P., & Scott, R. N. (1993). A new strategy for multifunction myoelectric control. IEEE Transactions on Biomedical Engineering, 40(1), 82–94. https://doi.org/10.1109/10.204774

Parajuli, N., Sreenivasan, N., Bifulco, P., Cesarelli, M., Savino, S., Niola, V., Esposito, D., Hamilton, T. J., Naik, G. R., Gunawardana, U., & Gargiulo, G. D. (2019). Real-Time EMG Based Pattern Recognition Control for Hand Prostheses: A Review on Existing Methods, Challenges and Future Implementation. Sensors (Basel, Switzerland), 19(20), 4596. https://doi.org/10.3390/s19204596










