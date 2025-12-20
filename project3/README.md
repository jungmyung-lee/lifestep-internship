# Project3
# EMG Preprocessing & Motion Synchronization

## Table of Contents
- [Overview](#overview)
- [Data Characteristics](#data-characteristics)
- [EMG Preprocessing Pipeline](#emg-preprocessing-pipeline)
- [EMG–Motion Synchronization](#emg–motion-synchronization)
- [Cutting Movement Analysis](#cutting-movement-analysis)
- [Scripts Overview](#scripts-overview)
  - [MATLAB Structure Exploration Tools (Initial Stage)](#matlab-structure-exploration-tools-initial-stage)
  - [EMG Preprocessing Pipelines (Signal Preparation Stage)](#emg-preprocessing-pipelines-signal-preparation-stage)
  - [EMG–Motion Synchronization and Visualization (Final Analysis Stage)](#emg–motion-synchronization-and-visualization-final-analysis-stage)
- [Output](#output)
- [Applications](#applications)
- [Requirements](#requirements)

---

## Overview

This module implements a complete EMG preprocessing and motion synchronization
pipeline for analyzing muscle activation in relation to 3D marker-based
movement data collected during biomechanical tasks such as squatting and
cutting.

The project was developed as part of a broader internship biomechanics
repository and focuses on transforming raw EMG signals into clean,
analysis-ready activation profiles, then synchronizing them with
motion-capture kinematics to enable interpretation of muscle–movement
relationships.

---

## Data Characteristics

- **Source:** Qualisys / QTM motion-capture system  
- **File format:** MATLAB `.mat`  
- **Data modalities:**
  - Five-channel EMG (analog signals)
  - 3D marker-based motion-capture data  
    (e.g., barbell markers and lower-body markers)
  - Frame-based kinematic information
  - Nested MATLAB structs and object arrays

---

## EMG Preprocessing Pipeline

Five raw EMG channels are processed using a standard biomechanics
preprocessing workflow:

- DC offset (baseline) removal  
- 60 Hz notch filtering  
- Band-pass filtering within the physiological EMG range  
- Full-wave rectification  
- RMS-based smoothing and envelope extraction  

These steps convert raw EMG recordings into clean, analysis-ready muscle
activation signals suitable for biomechanical interpretation and comparison
across movement phases.

---

## EMG–Motion Synchronization

Processed EMG envelopes are time-aligned with 3D marker trajectories,
including barbell markers and lower-body markers extracted from the
motion-capture data.

This synchronization enables direct comparison between muscle activation
patterns and movement phases.

**Example observations** include:
- Quadriceps activation peaking consistently near the bottom of each squat
  cycle

From this synchronized analysis, users can examine:
- Muscle activation timing relative to movement phases  
- Left–right asymmetry or compensation patterns  
- Movement efficiency and coordination  

---

## Cutting Movement Analysis

Although the repository primarily demonstrates squat-based examples, the same
EMG–motion synchronization workflow was also applied to cutting tasks during
the internship.

This allows analysis of muscle activation strategies in more dynamic,
multi-directional movements using the same preprocessing and synchronization
framework.

---

## Scripts Overview

This project is organized into modular scripts that together form an
end-to-end pipeline for EMG preprocessing and motion synchronization.
Each script corresponds to a specific stage of the analysis workflow,
ranging from initial data exploration to final synchronized visualization.

---

### MATLAB Structure Exploration Tools (Initial Stage)

These scripts are used at the **initial stage** of the pipeline to inspect and
understand the structure of MATLAB-based biomechanics datasets before any
signal processing or analysis is performed.

- **mat_struct_pretty_print.py**  
  Recursively prints the full structure of MATLAB `.mat` files in a
  human-readable format. This script is typically the first step used to
  identify where EMG channels, marker trajectories, and metadata are stored
  within deeply nested Qualisys/QTM exports.

- **interactive_mat_struct_browser.py**  
  An interactive exploration tool built with `ipywidgets`, allowing users to
  navigate MATLAB struct hierarchies through clickable fields and array
  indices. This script is used for detailed inspection once the overall
  structure has been identified.

---

### EMG Preprocessing Pipelines (Signal Preparation Stage)

These scripts implement the **core EMG preprocessing stage**, transforming
raw EMG signals into clean, analysis-ready activation profiles.

- **emg_single_preprocess.py**  
  Applies the full EMG preprocessing pipeline to a single EMG channel. This
  script is typically used early in the analysis to verify signal quality,
  confirm filter behavior, and validate preprocessing parameters before
  scaling to multiple channels.

- **emg_multi_preprocess_compare.py**  
  Extends the preprocessing workflow to multiple EMG channels and visualizes
  them in synchronized subplots. This script represents the finalized EMG
  preprocessing step prior to motion synchronization.

---

### EMG–Motion Synchronization and Visualization (Final Analysis Stage)

These scripts represent the **final stage** of the pipeline, where processed
EMG signals are synchronized with motion-capture data and visualized for
biomechanical interpretation.

- **emg_bar_trajectory_analysis.py**  
  Extracts and analyzes barbell marker trajectories to identify movement
  phases such as descent, bottom position, and ascent. This script provides
  the motion context required for EMG synchronization.

- **emg_bar_sync_plot.py**  
  Synchronizes a single EMG envelope with the Z-axis trajectory of a barbell
  marker, producing a compact visualization that highlights the temporal
  relationship between muscle activation and movement.

- **emg_bar_full_and_slice_visualization.py**  
  Combines multi-channel EMG preprocessing results with barbell trajectories
  and presents both full-duration and selected time-window views. This
  script serves as the final output visualization, enabling detailed
  inspection of muscle activation patterns across entire trials and specific
  movement segments.

---

## Output

The pipeline produces:
- Preprocessed EMG signals with rectified and envelope representations  
- Time-synchronized EMG–motion visualizations  
- Full-range and windowed plots for detailed biomechanical analysis  

---

## Applications

This pipeline can be applied to:
- Biomechanics and sports movement analysis  
- EMG signal quality inspection and preprocessing  
- Muscle–movement coordination studies  
- Analysis of squat and cutting movement strategies  
- Preprocessing for downstream modeling and machine learning  

---

## Requirements

- Python 3.8+  
- NumPy  
- SciPy  
- Matplotlib  
- ipywidgets  
