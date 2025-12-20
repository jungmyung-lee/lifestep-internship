# Project2  
# Automatic Gait Cycle Detection

## Table of Contents
- [Overview](#overview)
- [Input Data](#input-data)
- [Methodology](#methodology)
- [Hilbert Transform–Based Phase Analysis (Optional)](#hilbert-transformbased-phase-analysis-optional)
- [Interactive Visualization](#interactive-visualization)
- [Output](#output)
- [Applications](#applications)
- [Requirements](#requirements)

---

## Overview

This project presents a robust biomechanics-based algorithm for detecting  
**gait cycle initiation events** from **knee joint angle time-series data**.

Accurate gait cycle segmentation is a fundamental step in biomechanics,  
rehabilitation engineering, and human motion analysis. However, conventional  
methods based on simple peak or valley detection often fail when signals are  
noisy, asymmetric, or affected by irregular stride timing and speed changes.

This project implements an enhanced gait cycle detection pipeline using  
**knee flexion–extension trajectories** extracted from **OpenSim inverse  
kinematics (`.mot`) files**. Instead of relying on a single heuristic, the  
algorithm combines multiple physiologically meaningful constraints to ensure  
stable and reliable cycle detection.

Although this script focuses specifically on extracting gait cycle indices  
from knee angle signals, it is part of a broader biomechanics pipeline  
developed during my internship. In the full internship project, joint-angle  
extraction and movement-phase identification were performed using multiple  
variables provided in motion-capture datasets (e.g., `.mot` and `.mat` files),  
including knee, hip, and ankle angles, pelvis kinematics, ground reaction  
forces (GRF), and marker trajectories.

This script serves as a **modular component dedicated to robust gait-cycle  
segmentation**.

---

## Input Data

The input to this pipeline is an **OpenSim inverse kinematics output file  
(`.mot`)** containing joint kinematics over time.

### Data Characteristics
- **File format:** `.mot`
- **Source:** OpenSim Inverse Kinematics
- **Primary signal:** Knee flexion angle (left side)
- **Units:** Degrees
- **Sampling rate:** Automatically inferred from the time column

The script parses numeric rows from the `.mot` file and extracts:
- Time (seconds)
- Knee joint angle time-series

---

## Methodology

The algorithm detects gait cycle initiation events by identifying **valley  
points** in the knee angle trajectory, which correspond to the beginning of  
each gait cycle.

To ensure robustness under non-ideal conditions, the following steps are  
applied sequentially.

### 1. Signal Smoothing
A moving-average filter is applied to suppress high-frequency noise while  
preserving the global structure of the gait waveform.

### 2. Slope-Based Valley Detection
Candidate events are detected when the first derivative of the knee angle  
changes sign from negative to positive, indicating a local minimum.

### 3. Curvature Thresholding
A second-derivative (curvature) constraint is enforced to reject shallow or  
noise-induced minima, preventing false detections.

### 4. Local Window Refinement
Instead of trusting the raw zero-crossing location, the algorithm searches  
within a local temporal window to identify the true minimum of the knee angle.

### 5. Amplitude Filtering
Each candidate gait cycle is evaluated based on its angular amplitude. Stride  
segments that are too small or excessively large relative to the median  
gait-cycle amplitude are removed.

### 6. Length Filtering
Optional stride-length filtering removes gait cycles whose duration deviates  
significantly from the median stride length, helping handle irregular timing.

### 7. Minimum Distance Enforcement
A minimum inter-event distance constraint prevents detection of  
physiologically implausible, overly short gait cycles.

---

## Hilbert Transform–Based Phase Analysis (Optional)

In addition to time-domain valley detection, the script provides a  
**Hilbert transform–based phase analysis method**.

After band-pass filtering the knee angle signal, the analytic signal is  
computed and unwrapped to obtain instantaneous phase. Gait events are then  
detected at successive **2π** phase crossings, enabling cycle extraction  
based on continuous phase progression.

This approach is particularly useful for:
- relatively periodic gait patterns,
- phase-based gait modeling,
- validation and comparison with time-domain detection results.

---

## Interactive Visualization

An interactive visualization interface is provided using **ipywidgets**.

Users can dynamically adjust:
- slope and curvature thresholds,
- amplitude filtering bounds,
- stride length constraints.

Detected gait cycle initiation events are overlaid on the knee angle  
trajectory in real time, allowing intuitive inspection of algorithm behavior  
under different parameter settings.

---

## Output

The pipeline produces:
- knee angle time-series with detected gait cycle start indices,
- stable gait segmentation under noisy or irregular conditions,
- interactive plots for exploratory parameter tuning.

---

## Applications

This gait cycle detection pipeline can be applied to:
- gait analysis and rehabilitation research,
- synchronization of EMG and force-plate data,
- preprocessing for machine learning–based gait models,
- validation of inverse kinematics outputs,
- clinical biomechanics and assistive device studies.

---

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Pandas
- Matplotlib
- ipywidgets
