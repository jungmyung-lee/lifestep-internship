"""
Author: Jungmyung Lee

Automatic Gait Cycle Detection from Knee Angle Time-Series
----------------------------------------------------------
This script loads a .mot file containing joint kinematics and extracts gait cycle
events (valley points representing the start of each gait cycle) using an improved
biomechanics-based algorithm.

Purpose
-------
The goal is to detect gait initiation events robustly even when the knee angle
signal is noisy, asymmetric, or affected by inconsistent stride patterns.

Why This Method?
----------------
Knee angle trajectories typically exhibit a clear valley at the beginning of each
gait cycle. However, simple peak/valley detection often fails under noise,
slow walking, sudden speed changes, or irregular steps.
This script combines multiple criteria to ensure stable detection:

1) Slope-Based Valley Detection
   - Detects sign changes in the first derivative (negative → positive), which
     indicates a valley candidate.

2) Curvature Threshold (Second Derivative)
   - Prevents false valleys by requiring sufficient concavity.
   - Filters out shallow or noisy local minima.

3) Local Window Refinement
   - Instead of trusting the raw zero-crossing, the algorithm locates the
     true minimum within a neighborhood window.

4) Amplitude Filtering
   - Removes stride segments that are too small or excessively large relative
     to the median gait-cycle amplitude.
   - Ensures consistent kinematic quality even if the subject hesitates or
     drags a foot.

5) Length Filtering
   - Rejects gait cycles that are too short/long compared to the median stride length.
   - Helps handle irregular timing or noisy kinematic segments.

6) Minimum-Distance Enforcement
   - Ensures that no two gait starts occur unrealistically close together.

Why Hilbert Transform is Included
---------------------------------
The script also provides a Hilbert transform–based method for extracting phase
information from the knee signal. This allows cycle extraction based on 2π phase
progression and is useful for subjects with relatively periodic gait.

Interactive Visualization
-------------------------
ipywidgets sliders allow:
- adjusting slope / curvature thresholds
- amplitude & length filtering
- exploring how cycle detection changes in real time

"""

# ================================
# 0. Import libraries
# ================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# For interactive UI in Google Colab
!pip install -q ipywidgets
from ipywidgets import interact, Checkbox, FloatSlider

# ================================
# 1. Load .mot file
# ================================
filename = "/content/trial04_ik.mot"   # example

print("Loaded:", filename)

numeric_rows = []

with open(filename, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 0:
            continue
        try:
            float(parts[0])  # check if row is numeric
        except ValueError:
            continue
        numeric_rows.append(parts)

df = pd.DataFrame(numeric_rows, dtype=float)
print("Shape:", df.shape)

time = df.iloc[:, 0]
knee_l = df.iloc[:, 19]
knee_arr = knee_l.to_numpy().astype(float)

from scipy.signal import butter, filtfilt, hilbert

def hilbert_gait_cycles(time, sig, low=0.3, high=3.0, order=4):
    # ---------------------
    # 1) Bandpass filtering
    # ---------------------
    fs = 1.0 / np.median(np.diff(time))
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype="band")
    filt_sig = filtfilt(b, a, sig)

    # ---------------------
    # 2) Hilbert transform
    # ---------------------
    analytic = hilbert(filt_sig)
    phase = np.unwrap(np.angle(analytic))

    # ---------------------
    # 3) Find 2π phase crossings
    # ---------------------
    events = []
    phase0 = phase[0]

    total_phase = phase[-1] - phase0
    n_cycles = int(total_phase / (2*np.pi))

    for k in range(1, n_cycles+1):
        target = phase0 + k*2*np.pi
        idx = np.argmin(np.abs(phase - target))
        events.append(idx)

    return np.array(events), filt_sig, phase


# ================================
# 2. Gait cycle detection function
# ================================
# ================================
# 2. Improved gait cycle detection (enhanced version)
# ================================
def detect_gait_cycles(
    knee_arr,
    slope_eps=0.2,
    curvature_eps=0.3,  # Added curvature threshold parameter
    smooth_win=5,
    local_win=8,
    use_amp_filter=True,
    amp_min_factor=0.3,
    amp_max_factor=1.5,
    use_len_filter=False,
    len_factor=1.5
):

    # -----------------------------
    # A. Smoothing with moving average
    # -----------------------------
    smooth = np.convolve(knee_arr, np.ones(smooth_win)/smooth_win, mode='same')

    # Compute slope (1st derivative)
    slope = np.diff(smooth)

    # Compute curvature (2nd derivative)
    curvature = np.diff(slope)

    # -----------------------------
    # B. Find valley candidates using slope sign change
    # -----------------------------
    zero_crossings = []

    for i in range(1, len(slope)):
        # Ensure curvature index is valid
        if i-1 < len(curvature):

            # Valley condition: negative → positive slope AND curvature above threshold
            if slope[i-1] < -slope_eps and slope[i] > slope_eps and curvature[i-1] > curvature_eps:

                start = max(0, i - local_win)
                end   = min(len(knee_arr)-1, i + local_win)
                true_valley = start + np.argmin(knee_arr[start:end+1])

                zero_crossings.append(true_valley)

    zero_crossings = np.array(zero_crossings, dtype=int)

    # If too few valleys, return early
    if len(zero_crossings) < 2:
        return (np.array([], dtype=int),
                zero_crossings,
                np.array([]),
                np.array([]))

    # -----------------------------
    # C. Compute amplitude & step length for each interval
    # -----------------------------
    amplitudes = []
    lengths = []

    for k in range(len(zero_crossings) - 1):
        start = zero_crossings[k]
        end   = zero_crossings[k+1]
        seg = knee_arr[start:end+1]
        amplitudes.append(seg.max() - seg.min())
        lengths.append(end - start)

    amplitudes = np.array(amplitudes)
    lengths = np.array(lengths)

    valid_mask = np.ones(len(amplitudes), dtype=bool)

    # -----------------------------
    # D. Amplitude filtering
    # -----------------------------
    if use_amp_filter and len(amplitudes) > 0:
        median_amp = np.median(amplitudes)

        min_amp = median_amp * amp_min_factor
        max_amp = median_amp * amp_max_factor

        valid_mask &= (amplitudes >= min_amp)
        valid_mask &= (amplitudes <= max_amp)

    # -----------------------------
    # E. Length filtering
    # -----------------------------
    if use_len_filter and len(lengths) > 0:
        median_len = np.median(lengths)
        len_min = median_len / len_factor
        len_max = median_len * len_factor
        valid_mask &= (lengths >= len_min) & (lengths <= len_max)

    # -----------------------------
    # F. Minimum distance rule to prevent overly short cycles
    # -----------------------------
    if len(lengths) > 0:
        median_len_total = np.median(lengths)
        min_allowed = median_len_total * 0.5
        valid_mask &= (lengths >= min_allowed)

    # Final gait cycle start indices
    gait_cycle_start_indices = zero_crossings[:-1][valid_mask]

    return gait_cycle_start_indices, zero_crossings, amplitudes, lengths


# ================================
# 3. Plotting function
# ================================
def plot_knee_with_cycles(time, knee_l, gait_cycle_start_indices, title_suffix=""):
    plt.figure(figsize=(70, 4))
    plt.plot(time, knee_l, linewidth=1.2, label="Knee angle (left)")

    if len(gait_cycle_start_indices) > 0:
        ic_times = time.iloc[gait_cycle_start_indices]
        ic_angles = knee_l.iloc[gait_cycle_start_indices]
        plt.scatter(ic_times, ic_angles,
                    s=25, color='red', zorder=3, label='Gait cycle start')

    plt.legend()
    plt.title(f"Left Knee Angle {title_suffix}")
    plt.xlabel("Time (s)")
    plt.ylabel("Knee Angle (deg)")
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ================================
# 4. ipywidgets interactive UI
# ================================
@interact(
    use_amp_filter=Checkbox(value=True, description='Use amplitude filter'),
    amp_min_factor=FloatSlider(value=0.3, min=0.0, max=1.0, step=0.05),
    amp_max_factor=FloatSlider(value=1.5, min=1.0, max=5.0, step=0.1),

    use_len_filter=Checkbox(value=False, description='Use length filter'),
    len_factor=FloatSlider(value=1.5, min=1.0, max=3.0, step=0.1),
    slope_eps=FloatSlider(value=0.0, min=0.0, max=5.0, step=0.1),
    curvature_eps=FloatSlider(value=0.3, min=0.0, max=5.0, step=0.1)  # Added curvature slider
)
def update(use_amp_filter, amp_min_factor, amp_max_factor,
           use_len_filter, len_factor, slope_eps, curvature_eps):

    gait_indices, zero_crossings, amps, lens = detect_gait_cycles(
        knee_arr,
        slope_eps=slope_eps,
        curvature_eps=curvature_eps,
        use_amp_filter=use_amp_filter,
        amp_min_factor=amp_min_factor,
        amp_max_factor=amp_max_factor,
        use_len_filter=use_len_filter,
        len_factor=len_factor
    )

    print("Total zero-crossings:", len(zero_crossings))
    print("Detected gait starts:", len(gait_indices))

    if lens.size > 0:
        print(f"Length median: {np.median(lens):.2f}  (min={lens.min()}, max={lens.max()})")
    if amps.size > 0:
        print(f"Amp median: {np.median(amps):.2f}     (min={amps.min():.2f}, max={amps.max():.2f})")

    plot_knee_with_cycles(time, knee_l, gait_indices,
                          title_suffix=f"(eps={slope_eps})")
