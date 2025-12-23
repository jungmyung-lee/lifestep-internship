# Project5  
# Pose-Based Basketball Shooting Form Classification (GOOD / BAD)

---

![Cutting Motion](https://raw.githubusercontent.com/jungmyung-lee/lifestep-internship/main/13.gif)

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Dataset and Data Curation](#dataset-and-data-curation)
  - [Temporal Segmentation Strategy](#temporal-segmentation-strategy)
- [Input Data Specification](#input-data-specification)
- [Pose Estimation with YOLOv8-Pose](#pose-estimation-with-yolov8-pose)
- [Biomechanics-Informed Feature Design](#biomechanics-informed-feature-design)
- [Temporal Normalization](#temporal-normalization)

- [Model A: CNN + LSTM (Temporal Deep Learning Model)](#model-a-cnn--lstm-temporal-deep-learning-model)
  - [Why Use a 1D CNN](#why-use-a-1d-cnn)
  - [CNN Architecture Design](#cnn-architecture-design)
  - [Why Kernel Size = 5](#why-kernel-size--5)
  - [Why Use an LSTM](#why-use-an-lstm)
  - [Regularization and Training Strategy](#regularization-and-training-strategy)

- [Training Strategy and Optimization](#training-strategy-and-optimization)
  - [Why This Loss Function](#why-this-loss-function)
    - [Binary Cross-Entropy with Logits (BCEWithLogitsLoss)](#binary-cross-entropy-with-logits-bcewithlogitsloss)
    - [Why BCEWithLogitsLoss](#why-bcewithlogitsloss)
    - [Why Not Other Loss Functions](#why-not-other-loss-functions)
  - [Why This Optimizer](#why-this-optimizer)
    - [Why Adam](#why-adam)
    - [Why Learning Rate = 1e-3](#why-learning-rate--1e-3)
  - [Cross-Validation and Early Stopping](#cross-validation-and-early-stopping)
  - [Dropout](#dropout)
  - [Training Configuration Summary](#training-configuration-summary)

- [Model B: XGBoost (Feature-Based Machine Learning Model)](#model-b-xgboost-feature-based-machine-learning-model)
  - [Why a Feature-Based Model](#why-a-feature-based-model)
  - [Feature Vector Design](#feature-vector-design)
    - [Why Fixed-Length Vectors](#why-fixed-length-vectors)
    - [Why Preserve Channel Separation](#why-preserve-channel-separation)
  - [XGBoost Architecture and Hyperparameters](#xgboost-architecture-and-hyperparameters)
  - [Why XGBoost Performs Well in This Setting](#why-xgboost-performs-well-in-this-setting)

- [Evaluation Protocol](#evaluation-protocol)
- [Experimental Results](#experimental-results)
- [Limitations and Reflections](#limitations-and-reflections)
- [Conclusion](#conclusion)
- [Technologies Used](#technologies-used)
- [Author](#author)

 

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


### Temporal Segmentation Strategy (Manual Segmentation)

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

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/146a679e-9928-4f92-bd95-eb8927c53bd0" />

**The waveforms shown in this diagram are illustrative examples; the actual learned feature maps may exhibit different shapes depending on the data and training dynamics.**

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

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/5fa2f294-6ad3-4f32-a1f8-2fd99ad5b4cd" />

**The waveforms shown in this diagram are illustrative examples; the actual learned feature maps may exhibit different shapes depending on the data and training dynamics.**
---

### CNN Architecture Design

```python
nn.Conv1d(3, 32, kernel_size=5, padding=2)  
nn.ReLU()  
nn.MaxPool1d(2)  
```

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

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/4d243c36-12e0-4700-8d9e-abaf6744d84c" />

**The waveforms shown in this diagram are illustrative examples; the actual learned feature maps may exhibit different shapes depending on the data and training dynamics.**

---

#### Why Max Pooling

Max pooling serves multiple purposes:

- reduces **temporal resolution** while preserving salient motion patterns
- reduces **sensitivity to frame-level noise** and pose estimation jitter
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

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/929ff7b2-d838-43c5-9ad8-624d746a7c93" />


### LSTM Architecture Design

```python
nn.LSTM(
    input_size=32,
    hidden_size=64,
    batch_first=True
)
```

#### Hidden State Configuration

- The CNN outputs a **32-dimensional feature vector** per time step
- The LSTM uses a **hidden size of 64**

This expansion allows the model to:

- store richer **temporal context**
- encode **motion phase progression**
- integrate information across the full shooting sequence

Larger hidden sizes were intentionally avoided to reduce
**overfitting risk** under limited data conditions.

A hidden size of **64** provides sufficient expressive power
without excessive model complexity.

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/73fcaeaa-ecd6-46ed-8b5d-24d19bb6c0aa" />


---

#### Why Use Only the Final Hidden State

The task is **sequence-level classification** (GOOD / BAD),
not frame-level prediction.

- The **final hidden state** summarizes the entire shooting motion
- Intermediate hidden states are unnecessary once a global judgment is made

Each shooting sequence is treated as a **single coherent motion unit**,
rather than a collection of independent frames.

---

### Regularization and Training Strategy

---

## Training Strategy and Optimization

This section explains **how the CNN + LSTM model is trained**
and **why specific optimization choices were made**,
with a focus on **stability, generalization, and limited-data robustness**.

---

### Why This Loss Function?

#### Binary Cross-Entropy with Logits (BCEWithLogitsLoss)

The task is **binary sequence-level classification (GOOD / BAD)**.
The model outputs a **single scalar logit** representing shooting form quality.

Binary Cross-Entropy with logits is used:

```python
criterion = nn.BCEWithLogitsLoss()
```

### Why BCEWithLogitsLoss?

The model is trained using **Binary Cross-Entropy with Logits Loss
(`BCEWithLogitsLoss`)**.

**Key reasons:**

- Combines **sigmoid activation** and **binary cross-entropy** in a single operation  
- Avoids numerical instability caused by separately applying sigmoid  
- Strongly penalizes **confident but incorrect predictions**  
- Produces **well-calibrated decision boundaries** for binary classification  

This loss function is well suited for sequence-level binary classification
where **stable probability estimates** are required.

---

### Why Not Other Loss Functions?

- **Mean Squared Error (MSE)**  
  - Treats classification as regression  
  - Leads to poorly calibrated outputs and unstable decision thresholds  

- **Hinge Loss**  
  - Less stable under **small-sample regimes**  
  - Sensitive to margin selection  

- **Focal Loss**  
  - Designed for heavy class imbalance  
  - Introduces unnecessary complexity given the **balanced class distribution**

Overall, **BCEWithLogitsLoss** provides a **simple, stable, and principled objective**
for this problem setting.

---

### Why This Optimizer?

The model is optimized using the **Adam optimizer**:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### Why Adam?

The **Adam optimizer** is used for training the CNN + LSTM model.

**Key reasons:**

- Automatically adapts **learning rates per parameter**
- Handles:
  - heterogeneous feature scales
  - noisy gradients from pose estimation
- Well suited for **hybrid architectures (CNN + LSTM)**
- Empirically stable in **small-to-medium dataset regimes**

Overall, Adam offers a strong balance between **fast convergence**
and **robust optimization** for temporal deep learning models.

---

### Why Learning Rate = 1e-3?

- Provides **stable convergence** for the CNN–LSTM model
- Larger learning rates caused **unstable validation loss**
- Smaller learning rates:
  - slowed convergence
  - increased sensitivity to noise

A learning rate of **1e-3** achieves a practical balance between
**training speed** and **optimization stability**.

---

### Cross-Validation and Early Stopping

To ensure reliable performance estimation under limited data,
the following strategy is used:

- **Stratified 5-fold cross-validation**
  - Preserves class balance in each fold

- **Early stopping based on validation loss**
  - Prevents overfitting
  - Stops training when generalization no longer improves

---

### Dropout
Dropout is active **only during training**,  
and automatically disabled during evaluation and inference.

A dropout rate of **0.4** is applied before the final classification layer.

- The dataset size is limited (**N = 107**)
- Temporal deep models are prone to **overfitting** in such settings

Dropout encourages:

- robust feature utilization
- reduced reliance on any single temporal cue
- improved generalization across folds

### Training Configuration Summary

- **Optimizer:** Adam  
- **Learning rate:** 1e-3  
- **Maximum epochs:** 80  
- **Early stopping patience:** 8 epochs  
- **Classification threshold:** probability > 0.5  

This training strategy prioritizes **generalization and stability**
over aggressive optimization.


## Model B: XGBoost (Feature-Based Machine Learning Model)

This model represents a **classical machine learning approach**
that operates on **explicitly engineered, biomechanically interpretable features**.

Unlike the CNN + LSTM model, which learns representations directly from sequences,
XGBoost relies on **fixed-length feature vectors**
and focuses on learning **nonlinear decision boundaries** over structured inputs.

---

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/fb54737f-9a63-4227-8b03-da2261b52efd" />

**The waveforms shown in this diagram are illustrative examples; the actual learned feature maps may exhibit different shapes depending on the data and training dynamics.**

### Why a Feature-Based Model

The dataset size in this project is relatively small (**N = 107**),
which places strong constraints on the effectiveness of deep temporal models.

Tree-based ensemble methods such as XGBoost are known to:

- perform well in **small-to-medium data regimes**
- be robust to feature scaling and noise
- capture nonlinear feature interactions without requiring large datasets

XGBoost therefore serves as a **strong and realistic baseline**
for this problem setting.

---

### Feature Vector Design

Each shooting motion is flattened into a **240-dimensional feature vector**:

- **elbow angle trajectory** (80)
- **normalized wrist height** (80)
- **reference baseline signal** (80)

This fixed-length representation enables efficient tree-based learning
while preserving temporal progression implicitly.

---

#### Why Fixed-Length Vectors

- XGBoost requires **vectorized inputs**
- Temporal alignment is enforced through:
  - consistent motion start/end definition
  - resampling to a fixed length

Each shooting motion is represented as a **single point in feature space**,
allowing robust comparison across samples.

---

#### Why Preserve Channel Separation

Each biomechanical signal is kept as a **separate contiguous block**
within the feature vector.

This allows XGBoost to:

- independently evaluate elbow- and wrist-related motion patterns
- learn cross-channel interactions through tree splits
- retain interpretability at the signal level

---

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/09f05dcd-5e8b-485d-bbf8-25b207668edf" />


### XGBoost Architecture and Hyperparameters

```python
from xgboost import XGBClassifier

XGBClassifier(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.02,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)
```

The XGBoost classifier is configured with the following principles:

- **Shallow trees (max_depth = 3)** to enforce strong regularization
- **Many estimators (n_estimators = 400)** to stabilize ensemble behavior
- **Low learning rate (learning_rate = 0.02)** for gradual optimization
- **Subsampling (subsample = 0.9, colsample_bytree = 0.9)** to reduce variance

These choices prevent memorization,
encourage smooth decision boundaries,
and improve generalization under limited data.

---

### Why XGBoost Performs Well in This Setting

XGBoost benefits from several factors in this project:

- domain-informed feature design
- explicit temporal alignment
- **compact but but informative representations**
- built-in regularization mechanisms

As a result, XGBoost exhibits **more stable generalization**
than deeper temporal models whose representational capacity
cannot be fully exploited with limited data.

---

## Evaluation Protocol

- **Stratified 5-fold cross-validation**

- Metrics:
  - accuracy
  - F1-score

- Analysis:
  - confusion matrix

All models are evaluated using the same protocol
to ensure fair comparison.

---

## Experimental Results

| Model | Accuracy (mean ± std) | F1-score (mean ± std) |
|------|----------------------|-----------------------|
| CNN + LSTM | 0.66 ± 0.07 | 0.64 ± 0.08 |
| XGBoost | 0.74 ± 0.04 | 0.73 ± 0.05 |

XGBoost demonstrates **higher mean performance**
and **lower variance** across folds.

---

## Limitations and Reflections

- The dataset size is **relatively small**
- Deep temporal models are sensitive to **data scarcity**
- Feature design plays a critical role in model effectiveness

In this project, **feature-based machine learning**
proved more reliable than **end-to-end temporal deep learning**
under constrained data conditions.

---

## Conclusion

This project analyzed pose-based basketball shooting motions
using real instructional class videos.

By integrating:

- basketball coaching experience
- biomechanical domain knowledge
- pose estimation, machine learning, and deep learning

shooting motions were transformed into
interpretable temporal representations.

Experimental results suggest that, **under limited data conditions**,
the **XGBoost machine learning model** can exhibit more stable performance
than the **CNN + LSTM deep learning model**.

This finding highlights the importance of
**aligning model complexity with data regime**
when designing motion analysis systems.

---

## Technologies Used

### Programming Languages and Libraries
- Python
- PyTorch
- OpenCV
- scikit-learn
- XGBoost

### Models and Architectures
- Convolutional Neural Networks (1D CNN)
- Long Short-Term Memory Networks (LSTM)
- YOLOv8-Pose (2D human pose estimation)


---

## Author

**Jungmyung Lee**  
Developed independently during the **LifeStep internship**  
Over **four years of basketball coaching experience**


