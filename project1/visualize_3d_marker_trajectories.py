"""
Author: Jungmyung Lee

3D Marker Trajectory Inspection & Static Snapshot
-------------------------------------------------
This script loads a Qualisys-style 3D motion-capture file (e.g., cutting.mat)
and visualizes all labeled markers in two ways:

1) Full 3D trajectories for each marker across time
2) A static snapshot of the last valid 3D position for every marker

Main processing steps:
1) Load a .mat mocap file and extract the main struct
2) Robustly locate the Trajectories → Labeled → {Labels, Data} block
3) Filter samples using the residual channel (if available)
4) Optionally downsample frames for faster visualization
5) Plot 3D trajectories for all markers
6) Plot the final valid 3D positions as a scatter plot

This script is mainly used for:
- sanity-checking raw motion-capture data,
- verifying marker naming and labeling consistency,
- quickly assessing data quality before building more complex pipelines.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import scipy.io as sio

# -- 1) Load .mat file (Colab upload or local path) --
try:
    from google.colab import files
    IN_COLAB = True
except Exception:
    IN_COLAB = False

if IN_COLAB:
    # In Colab, open a file-upload dialog and use the first uploaded file.
    uploaded = files.upload()
    path = list(uploaded.keys())[0]
else:
    # Local / non-Colab environment: set the path manually.
    path = "cutting.mat"

print("Loaded file:", path)

# -- 2) Load MAT struct and extract Trajectories → Labeled block --
# scipy.io.loadmat returns a dict with keys like "__header__", "__version__", "__globals__"
# plus one main user key (e.g., "MotionData"). We ignore the metadata keys.
data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
user_keys = [k for k in data.keys() if not (k.startswith("__") and k.endswith("__"))]
top_key = user_keys[0]
S = data[top_key]


def get_labeled_block(S):
    """
    Extract labeled marker names and 3D trajectories from various Qualisys/QTM layouts.

    Returns:
        labels: list of marker names (str)
        arr:    np.ndarray of shape [markers, 4, frames] with (x, y, z, residual)
    """
    # Case A: MATLAB-like struct with fields, e.g. S.Trajectories.Labeled.Data
    if hasattr(S, "Trajectories"):
        T = S.Trajectories
        L = T.Labeled if hasattr(T, "Labeled") else None
        if L is not None and hasattr(L, "Data"):
            raw_labels = L.Labels
            labels = (
                list(raw_labels)
                if isinstance(raw_labels, (list, tuple, np.ndarray))
                else [raw_labels]
            )
            arr = np.array(L.Data)
            labels = [str(x) for x in labels]
            return labels, arr

    # Case B: numpy structured array with something like dtype [("Labeled","O"), ...]
    if isinstance(S, (np.void, np.ndarray)):
        try:
            T = S["Trajectories"][()] if isinstance(S, np.void) else S["Trajectories"]
        except Exception:
            T = S

        if isinstance(T, np.ndarray) and T.dtype.names and "Labeled" in T.dtype.names:
            L = T["Labeled"][0, 0]
            labels_raw = L["Labels"][0, 0]
            data_raw = L["Data"][0, 0]

            if isinstance(labels_raw, np.ndarray):
                labels = [
                    str(x.item() if hasattr(x, "item") else x)
                    for x in labels_raw.ravel()
                ]
            else:
                labels = [str(labels_raw)]

            arr = np.array(data_raw)
            return labels, arr

    raise RuntimeError("Could not find Trajectories → Labeled → Data in this file.")


# Extract labels and marker trajectories
labels, arr = get_labeled_block(S)
print(f"Markers: {len(labels)} | Data shape: {arr.shape} (expected: [markers, 4, frames])")
frames = arr.shape[2]

# -- 3) Visualization settings --
residual_threshold = 2.0   # keep samples with residual ≤ threshold; set None to disable
downsample = 1             # e.g., 2 means take every 2nd frame


# -- 4) Helper: enforce equal axes for 3D plots --
def set_equal_3d(ax):
    """Set 3D axes to a cubic box so that x, y, z scales are comparable."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    xmid = np.mean(xlim)
    ymid = np.mean(ylim)
    zmid = np.mean(zlim)

    max_range = max(
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        zlim[1] - zlim[0],
    )
    r = max_range / 2.0

    ax.set_xlim3d([xmid - r, xmid + r])
    ax.set_ylim3d([ymid - r, ymid + r])
    ax.set_zlim3d([zmid - r, zmid + r])


# -- 5) Plot full 3D trajectories for all markers --
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

for i, name in enumerate(labels):
    xyzr = arr[i, :, :]
    x, y, z, r = xyzr[0], xyzr[1], xyzr[2], xyzr[3]

    idx = np.arange(frames)

    # Filter by residual if available; otherwise, just require finite x/y/z.
    if residual_threshold is not None:
        mask = np.isfinite(r) & (r <= residual_threshold)
    else:
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    # Optional temporal downsampling
    if downsample > 1:
        mask[~((idx % downsample) == 0)] = False

    if np.count_nonzero(mask) < 2:
        continue

    ax.plot(x[mask], y[mask], z[mask], linewidth=1)

ax.set_title("All Markers: 3D Trajectories")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
set_equal_3d(ax)
plt.show()

# -- 6) Plot last valid 3D position for each marker --
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

for i, name in enumerate(labels):
    xyzr = arr[i, :, :]
    x, y, z, r = xyzr[0], xyzr[1], xyzr[2], xyzr[3]

    if residual_threshold is not None:
        mask = np.isfinite(r) & (r <= residual_threshold)
    else:
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    if np.any(mask):
        # Take the index of the last valid sample for this marker
        k = np.where(mask)[0][-1]
        ax.scatter(x[k], y[k], z[k], s=10)

ax.set_title("All Markers: Last Valid Positions")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
set_equal_3d(ax)
plt.show()
