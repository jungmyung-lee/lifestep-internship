# Pose-Based Basketball Shooting Form Classification (GOOD / BAD)

---

## Table of Contents

- Project Overview  
- Motivation  
- Dataset and Data Curation  
- Input Data Specification  
- Pose Estimation with YOLOv8-Pose  
- Biomechanics-Informed Feature Design  
- Temporal Normalization  
- Model A: CNN + LSTM (Temporal Deep Learning Model)  
  - Why Use a 1D CNN  
  - CNN Architecture Design  
  - Why Use an LSTM  
  - Regularization and Training Strategy  
- Model B: XGBoost (Feature-Based Machine Learning Model)  
  - Why a Feature-Based Model  
  - Feature Vector Design  
  - XGBoost Architecture and Hyperparameters  
- Evaluation Protocol  
- Experimental Results  
- Limitations and Reflections  
- Conclusion  
- Technologies Used  
- Author  

---

## Project Overview

This repository presents a **pose-based basketball shooting form classification pipeline**
developed independently during the **LifeStep internship**.

The system analyzes **real basketball class session videos** and classifies each
shooting motion as **GOOD** or **BAD** using pose-derived temporal features.

Rather than treating this as a generic video classification task,
the pipeline is intentionally designed to align:

- basketball domain knowledge  
- biomechanical interpretability  
- machine learning model capacity  

The goal is not only classification accuracy,
but also **meaningful motion understanding under limited data conditions**.

---

## Motivation

Basketball shooting is a **time-dependent coordinated motion**
that involves precise sequencing of joints rather than isolated postures.

The data used in this project originates from **real basketball class sessions**
recorded at the **LJM Basketball Academy**, previously operated by the author.

Through years of coaching experience, the following question repeatedly arose:

**Can basketball shooting form differences be explained quantitatively,
rather than relying solely on intuition?**

During the LifeStep internship, the author gained practical and theoretical experience in:

- pose estimation  
- temporal modeling  
- machine learning and deep learning system design  

In addition, the author studied **full-body anatomy, joint function, and gait cycles**,
enabling basketball shooting motions to be interpreted as structured biomechanical processes.

This project integrates coaching experience and AI training
to formalize basketball shooting analysis as a **pose-based temporal classification problem**.

---

## Dataset and Data Curation

- Data source: real basketball class session videos  
- Total samples: **107 shooting videos**  
- Label classes:
  - **GOOD**: mechanically stable and efficient shooting form  
  - **BAD**: unstable or inefficient shooting form  

All labels were **manually annotated by the author**,
who has **over four years of basketball coaching experience**.

### Temporal Segmentation Strategy

No traditional preprocessing (e.g., filtering or smoothing) was applied.
Instead, data quality was ensured through **strict temporal segmentation**.

- Start point: defined as the **deep motion initiation** of the shooting action  
- End point: defined as the moment **just after ball release, before postural breakdown**  

All 107 videos were curated using this same temporal definition,
ensuring consistent motion alignment across samples.

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

---

## Pose Estimation with YOLOv8-Pose

Human pose keypoints are extracted using **YOLOv8-Pose**.

- Output: 2D body keypoints per frame  
- Key joints used in this project:
  - shoulder  
  - elbow  
  - wrist  
  - hip  

Only the **primary detected person** is used in each frame,
ensuring consistency across sequences.

---

## Biomechanics-Informed Feature Design

From pose keypoints, three interpretable time-series features are derived:

- elbow joint angle trajectory  
- normalized wrist vertical position (relative to torso length)  
- reference baseline signal  

These features are intentionally selected to reflect
key biomechanical aspects of basketball shooting mechanics.

Feature design and model structure are **deliberately aligned**
so that the model learns **meaningful motion patterns rather than arbitrary correlations**.

---

## Temporal Normalization

Each shooting motion varies in duration.
To enable consistent modeling:

- all time-series are resampled to **80 time steps**  
- z-score normalization is applied per sequence  

This enforces temporal alignment while preserving relative motion dynamics.

---

## Model A: CNN + LSTM (Temporal Deep Learning Model)

This model learns **temporal motion patterns directly**
from pose-derived time-series features.

The architecture follows a two-stage design:

1. a **1D CNN encoder** for local temporal pattern extraction  
2. an **LSTM** for sequence-level temporal modeling  

---

### Why Use a 1D CNN

A 1D CNN operating along the **temporal axis**
is well suited for pose-based motion analysis.

Basketball shooting contains localized temporal patterns such as:

- rapid elbow extension near release  
- short wrist acceleration bursts  
- brief stabilization phases  

The CNN acts as a **temporal feature encoder**,
emphasizing motion structure while suppressing frame-level noise.

---

### CNN Architecture Design

```python
nn.Conv1d(3, 32, kernel_size=5, padding=2)
nn.ReLU()
nn.MaxPool1d(2)

#### Design Rationale

- Convolution is applied **only along the temporal axis**
- The input channels are already **semantically meaningful**:
  - **elbow joint angle**
  - **normalized wrist height**
  - **reference baseline**
- Inter-joint relationships are embedded at the **feature design stage**
- Avoiding spatial convolutions reduces **unnecessary model complexity**

This design focuses the model on **how motion evolves over time**,  
rather than **where joints are located spatially**.

---

### Why Kernel Size = 5

A kernel size of **5 frames** captures:

- the **current time step**
- **short-range temporal context** before and after

This is sufficient to model:

- **elbow extension transitions** near the release phase
- **wrist elevation changes**
- **brief stabilization or hesitation patterns**

Comparison:

- **kernel size = 1**
  - behaves like a **point-wise projection**
  - ignores **temporal continuity**

- **kernel size ≥ 7**
  - increases **parameter count**
  - blurs **fine-grained temporal transitions**

Kernel size **5** provides a balanced trade-off between  
**temporal expressiveness** and **training stability**.

---

#### Why 32 Filters

The input consists of **three biomechanically interpretable channels**.

Using **32 filters** allows the CNN to learn:

- different combinations of **elbow–wrist coordination**
- **phase-specific motion signatures**
- **subject-invariant local temporal patterns**

Using significantly more filters would:

- increase **overfitting risk** under limited data conditions
- provide **diminishing returns** in representation quality

Thus, **32 filters** represent a **controlled and sufficient expansion**
of representational capacity.

---

#### Why Max Pooling

Max pooling serves multiple purposes:

- reduces **temporal resolution** while preserving salient motion patterns
- suppresses **frame-level noise** and pose estimation jitter
- shortens the sequence length passed to the LSTM

This encourages the model to focus on **motion trends**
rather than **exact frame timing**,
and improves **training stability**.

---

### Why Use an LSTM

While the CNN extracts **local temporal motion patterns**,
it does not capture **how those patterns are ordered over time**.

Basketball shooting is inherently a **sequence-level motion process**:

- preparation  
- upward motion  
- release  
- follow-through  

Correct timing and phase transitions are often more important
than isolated posture quality.

An LSTM is therefore used to model:

- **long-range temporal dependencies**
- **phase transitions across the entire shooting motion**
- **cumulative motion consistency**


