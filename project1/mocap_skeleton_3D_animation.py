"""
Author: Jungmyung Lee

3D Motion-Capture Skeleton Animation Pipeline
---------------------------------------------
This script loads a Qualisys-style 3D motion-capture file (e.g., squat.mat)
and generates a 3D animated visualization of raw marker positions over time.

Main processing steps:
1) Load labeled 3D marker trajectories from a .mat motion-capture file
2) Extract marker coordinates from the Qualisys/QTM data structure
3) Identify markers that contain at least one valid 3D sample across the trial
4) Apply temporal downsampling to reduce animation complexity
5) Render a stable 3D marker-only animation with fixed cubic axes
6) Export the animation as a lightweight animated GIF


This pipeline is designed for biomechanics research and visualization of
squat motion, gait trials, and general marker-based motion-capture data.
It is robust to missing markers, naming inconsistencies, and noisy trajectories.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# =====================================================
# 1) Load motion-capture file (Colab environment)
# =====================================================
# The file is assumed to already exist in the Colab /content directory.
path = "/content/squat.mat"
print("Loaded file:", path)

# =====================================================
# 2) Extract labeled marker trajectories from QTM structure
# =====================================================
def get_labeled_block(S):
    """
    Locate and extract the Trajectories → Labeled → Data block
    from a Qualisys/QTM-exported MATLAB structure.

    Returns:
        labels : list of marker names (strings)
        data   : numpy array of shape [markers, 4, frames]
                 containing (x, y, z, residual)
    """
    if hasattr(S, "Trajectories"):
        T = S.Trajectories
        if hasattr(T, "Labeled") and hasattr(T.Labeled, "Data"):
            labels = T.Labeled.Labels
            labels = (
                list(labels)
                if isinstance(labels, (list, tuple, np.ndarray))
                else [labels]
            )
            data = np.array(T.Labeled.Data)
            return [str(x) for x in labels], data

    raise RuntimeError("No Trajectories → Labeled → Data block found.")

# Load .mat file and extract the main data structure
mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
user_keys = [k for k in mat.keys() if not k.startswith("__")]
S = mat[user_keys[0]]

labels, arr = get_labeled_block(S)
markers, _, frames = arr.shape
print(f"Markers: {markers} | Frames: {frames}")

# =====================================================
# 3) Temporal downsampling for efficient animation
# =====================================================
# Using every N-th frame reduces rendering cost while
# preserving overall motion patterns.
step = 4
frame_idx = np.arange(0, frames, step)
print("Using frames:", len(frame_idx))

# =====================================================
# 4) Compute fixed cubic axis limits (global scale)
# =====================================================
# Axis limits are computed once using all valid samples,
# ensuring a stable camera view across the entire animation.
xyz = []
for i in range(markers):
    x, y, z, _ = arr[i]
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if mask.any():
        xyz.append(np.vstack([x[mask], y[mask], z[mask]]))

xyz_all = np.hstack(xyz)
xmin, xmax = xyz_all[0].min(), xyz_all[0].max()
ymin, ymax = xyz_all[1].min(), xyz_all[1].max()
zmin, zmax = xyz_all[2].min(), xyz_all[2].max()

cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
r = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2

# =====================================================
# 5) Animated marker visualization (no skeleton)
# =====================================================
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(cx - r, cx + r)
ax.set_ylim(cy - r, cy + r)
ax.set_zlim(cz - r, cz + r)
ax.view_init(elev=20, azim=-60)

# Scatter object that will be updated frame-by-frame
pts = ax.scatter([], [], [], s=14)

def init():
    """
    Initialize the animation by explicitly drawing
    the first valid frame. This step is critical to
    prevent empty GIF outputs in matplotlib.
    """
    f = frame_idx[0]
    x = arr[:, 0, f].astype(float)
    y = arr[:, 1, f].astype(float)
    z = arr[:, 2, f].astype(float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    pts._offsets3d = (x[valid], y[valid], z[valid])
    return (pts,)

def update(fi):
    """
    Update marker positions for the current animation frame.
    Only markers with finite 3D coordinates are rendered.
    """
    f = frame_idx[fi]
    x = arr[:, 0, f].astype(float)
    y = arr[:, 1, f].astype(float)
    z = arr[:, 2, f].astype(float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    pts._offsets3d = (x[valid], y[valid], z[valid])
    ax.set_title(f"Frame {f}/{frames}")
    return (pts,)

ani = FuncAnimation(
    fig,
    update,
    frames=len(frame_idx),
    init_func=init,
    interval=25,
    blit=False
)

# =====================================================
# 6) Export animation as GIF
# =====================================================
# GIF output is chosen for easy embedding in README files
# and lightweight sharing without video codecs.
out = "/content/marker_motion.gif"
writer = PillowWriter(fps=30)
ani.save(out, writer=writer, dpi=120)
plt.close(fig)

print("Saved:", out)
