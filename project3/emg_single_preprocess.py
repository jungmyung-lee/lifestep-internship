"""
Author: Jungmyung Lee

One EMG Preprocessing Pipeline (Single-Channel Analysis)
---------------------------------------------------
This script loads *one* EMG channel from a MATLAB .mat motion-capture file
and generates a single MATLAB-style plot for that channel after applying a
standard biomechanics preprocessing workflow:

1) Baseline (DC offset) removal
2) 100 Hz notch filtering to reduce powerline and motion artifacts
3) Bandpass filtering between 20–450 Hz (physiological EMG band)
4) Full-wave rectification
5) RMS envelope extraction using a 50 ms moving window

The script produces a figure showing both the rectified EMG signal and its
smoothed RMS envelope. This single-channel visualization is commonly used
for inspecting EMG quality in gait, squat, and sports biomechanics data.
"""


import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

# =========================
# Load EMG 3
# =========================
mat = sio.loadmat("/content/squat.mat")
root = mat['qtm_50_c2']
analog = root[0,0]['Analog']
emg3 = analog[0,5]

raw = emg3['Data'][0]
fs = float(emg3['Frequency'][0,0])

# =========================
# Preprocessing pipeline
# =========================

# 1. Baseline removal
x = raw - np.mean(raw)

# 2. Notch filter 95–105 Hz
b_notch, a_notch = iirnotch(100/(fs/2), 30)
x = filtfilt(b_notch, a_notch, x)

# 3. Bandpass 20–450 Hz
def bandpass(low, high, fs, order=4):
    nyq = fs * 0.5
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return b, a

b_bp, a_bp = bandpass(20, 450, fs)
x_bp = filtfilt(b_bp, a_bp, x)

# 4. Rectification
rect = np.abs(x_bp)

# 5. RMS envelope (50 ms)
win = int(0.05 * fs)
window = np.ones(win) / win
env = np.sqrt(np.convolve(rect**2, window, mode='same'))

# Time vector
t = np.arange(len(raw)) / fs

# =========================
# PLOT (MATLAB style)
# =========================
plt.figure(figsize=(14,5))

plt.plot(t, rect, color='tab:blue', linewidth=0.6, label="Rectified")
plt.plot(t, env, color='tab:orange', linewidth=1.5, label="Envelope")

plt.title("EMG 3 – Rectified + Envelope", fontsize=16)
plt.xlabel("time (s)", fontsize=14)
plt.ylabel("voltage (V)", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()
