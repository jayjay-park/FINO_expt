import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import cm

initial_guess = "smooth"
norm = "infty"
max_iter = 1999
folder = "."

h5_jac_200 = f"{folder}/inversion_history_JAC_200_{initial_guess}.h5"
h5_mse = f"{folder}/inversion_history_MSE_{initial_guess}.h5"
h5_devito = f"{folder}/inversion_history_Devito_{initial_guess}.h5"

# Load paths from HDF5 files
with h5py.File(h5_jac_200, "r") as f:
    path_ngd = f["a"][:]  # shape: [1, T, H, W]

with h5py.File(h5_mse, "r") as f:
    path_gd = f["a"][:]

with h5py.File(h5_devito, "r") as f:
    path_d = f["a"][:]

# Flatten all: remove batch and reshape to [T, -1]
X_ngd = path_ngd.squeeze(0).reshape(path_ngd.shape[1], -1)
X_gd = path_gd.squeeze(0).reshape(path_gd.shape[1], -1)
X_d = path_d.squeeze(0).reshape(path_d.shape[1], -1)

# Subsample every 200 iterations
subsample = slice(0, None, 20)
X_ngd = X_ngd[subsample]
X_gd = X_gd[subsample]
X_d = X_d[subsample]

print("NGD path shape (after squeeze and slice):", X_ngd.shape)
print("GD path shape (after squeeze and slice):", X_gd.shape)
print("Devito path shape (after squeeze and slice):", X_d.shape)

# Stack together to get common 2D subspace
X = np.vstack([X_d]) #X_gd,
X_mean = X.mean(axis=0)
X_centered = X - X_mean

# PCA directions
U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)
v1, v2 = Vh[:2]

# Project all trajectories
coords_ngd = np.stack([(X_ngd - X_mean) @ v1, (X_ngd - X_mean) @ v2], axis=1)
coords_gd = np.stack([(X_gd - X_mean) @ v1, (X_gd - X_mean) @ v2], axis=1)
coords_d = np.stack([(X_d - X_mean) @ v1, (X_d - X_mean) @ v2], axis=1)

# Plotting
plt.figure(figsize=(8, 6))
T = coords_gd.shape[0]
t_vals = np.arange(T)

from matplotlib import cm
from matplotlib.colors import Normalize

# Use only the darker part of each colormap by restricting the range
cmap_blues = cm.Blues_r  # reversed so darker values come first
cmap_reds = cm.Reds_r
cmap_greens = cm.Greens_r

norm = Normalize(vmin=0, vmax=T)  # Normalize t_vals

plt.figure(figsize=(8, 6))

plt.scatter(coords_gd[:, 0], coords_gd[:, 1],
            c=t_vals, cmap=cmap_reds, norm=norm, s=8, label='MSE')
plt.scatter(coords_ngd[:, 0], coords_ngd[:, 1],
            c=t_vals, cmap=cmap_blues, norm=norm, s=8, label='Jvp (FIM: 200)')
plt.scatter(coords_d[:, 0], coords_d[:, 1],
            c=t_vals, cmap=cmap_greens, norm=norm, s=8, label='Numerical Simulator')


# Mark start and end points
# plt.plot(coords_gd[0, 0], coords_gd[0, 1], 'o', color='navy', label='GD Start', markersize=6)
# plt.plot(coords_gd[-1, 0], coords_gd[-1, 1], 'X', color='black', label='GD End', markersize=6)
# plt.plot(coords_ngd[0, 0], coords_ngd[0, 1], 'o', color='darkred', label='NGD Start', markersize=6)
# plt.plot(coords_ngd[-1, 0], coords_ngd[-1, 1], 'X', color='black', markersize=6)
# plt.plot(coords_d[0, 0], coords_d[0, 1], 'o', color='darkgreen', label='Sim Start', markersize=6)
# plt.plot(coords_d[-1, 0], coords_d[-1, 1], 'X', color='black', markersize=6)

plt.xlabel('Direction 1')
plt.ylabel('Direction 2')
plt.title('Optimization Trajectories (Projected to 2D)')
# plt.colorbar(label='Iteration (every 10 steps)', orientation='vertical')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("trajectory_comparison_colored.png", dpi=150)
plt.close()
