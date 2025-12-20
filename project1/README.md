# 3D Motion-Capture Visualization & Skeleton Reconstruction Pipeline

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data Format (Qualisys / QTM)](#data-format-qualisys--qtm)
- [Scripts](#scripts)
  - [visualize_3d_marker_trajectory.py](#visualize_3d_marker_trajectorypy)
  - [reconstruct_static_skeleton.py](#reconstruct_static_skeletonpy)
  - [mocap_skeleton_3d_animation.py](#mocap_skeleton_3d_animationpy)
- [Execution Order](#execution-order)
- [Key Design Decisions](#key-design-decisions)
- [Output Examples](#output-examples)
- [Applications](#applications)
- [Requirements](#requirements)

---

## Overview

This repository contains a set of Python scripts for inspecting, visualizing,
and reconstructing human motion from Qualisys/QTM-style 3D motion-capture data.

The pipeline is designed for biomechanics-oriented motion analysis and focuses on:
- robust handling of missing or noisy markers,
- flexible parsing of multiple QTM export formats,
- clear marker-level inspection prior to advanced modeling.

Rather than applying a predefined biomechanical model,
the pipeline emphasizes direct visualization and validation
of labeled 3D marker trajectories.

---

## Project Structure

The repository consists of three standalone Python scripts,
each addressing a different stage of motion-capture data analysis:

├── visualize_3d_marker_trajectory.py
├── reconstruct_static_skeleton.py
└── mocap_skeleton_3D_animation.py

yaml
코드 복사

Each script can be executed independently depending on the analysis objective.

---

## Data Format (Qualisys / QTM)

All scripts take Qualisys/QTM-exported MATLAB (`.mat`) files as input
and extract labeled 3D marker trajectories for processing and visualization.

The input data structure contains:
- Trajectories → Labeled → Labels
- Trajectories → Labeled → Data

The `Data` array follows the convention:
- Shape: `[markers, 4, frames]`
- Channel 0: X coordinate
- Channel 1: Y coordinate
- Channel 2: Z coordinate
- Channel 3: Residual (tracking quality)

The parsing logic is designed to tolerate:
- missing markers,
- NaN segments,
- minor variations in QTM export structure.

---

## Scripts

### visualize_3d_marker_trajectory.py

**3D Marker Trajectory Inspection & Static Snapshot**

This script is intended for initial inspection of raw motion-capture data.

Main features:
- robust extraction of labeled marker trajectories,
- optional residual-based filtering,
- full 3D trajectory visualization for all markers,
- static visualization of the last valid 3D position per marker.

Typical use cases:
- verifying marker labeling consistency,
- identifying tracking dropouts or noisy segments,
- assessing overall data quality prior to reconstruction or modeling.

---

### reconstruct_static_skeleton.py

**Marker-Based Skeleton Reconstruction (Static Frame)**

This script reconstructs a biomechanical stick-figure representation
from labeled marker trajectories.

Key characteristics:
- prefix-based marker name matching to handle naming variations,
- automatic selection of the last valid 3D sample per marker,
- residual-aware filtering to improve reconstruction reliability,
- anatomical reconstruction of pelvis, torso, limbs, and feet,
- equalized 3D axes for accurate spatial interpretation.

This script is primarily used to:
- validate anatomical marker placement,
- inspect reconstruction consistency,
- debug marker definitions before generating full animations.

---

### mocap_skeleton_3d_animation.py

**3D Marker Animation Pipeline**

This script generates a time-resolved 3D animation of labeled markers.

Main processing steps:
- extraction of valid marker samples across time,
- temporal downsampling (using every N-th frame) to reduce rendering cost,
- global axis scaling to maintain a stable camera view,
- marker-only 3D animation without imposed skeleton constraints,
- export of the animation as an animated GIF.

The resulting animation supports qualitative inspection
of motion patterns across time.

---

## Execution Order

1. Run `visualize_3d_marker_trajectory.py`  
   → Inspect raw trajectories and identify problematic markers

2. Run `reconstruct_static_skeleton.py`  
   → Verify anatomical consistency and marker mapping

3. Run `mocap_skeleton_3d_animation.py`  
   → Generate a clean 3D animation for temporal inspection

---

## Key Design Decisions

- Marker-level visualization is prioritized to enable
  direct inspection of raw motion-capture data.

- Residual-based filtering is applied when available
  to reduce the impact of tracking artifacts.

- Fixed global axis limits are enforced to ensure
  a consistent spatial reference across frames.

- Each script is designed to operate independently
  to support flexible exploratory analysis.

---

## Output Examples

- 3D marker trajectory plots
- Static stick-figure reconstructions
- Marker-based 3D animations (GIF)

---

## Applications

This pipeline can be applied to:
- general biomechanics motion analysis,
- gait and rehabilitation studies,
- motion-capture data validation,
- integration with EMG or force-plate pipelines,
- preprocessing for machine learning or inverse kinematics models.

---

## Requirements

- Python 3.8
- NumPy
- SciPy
- Matplotlib
