import numpy as np
import matplotlib.pyplot as plt

# Load paths
path_ngd = np.load("posterior_path_ngd.npy")   # shape: [T, H, W]
path_gd = np.load("posterior_path_gd.npy")     # shape: [T, H, W]

# Flatten all
X_ngd = path_ngd.reshape(path_ngd.shape[0], -1)
X_gd = path_gd.reshape(path_gd.shape[0], -1)

# Stack together to get common 2D subspace
X = np.vstack([X_ngd, X_gd])
X_mean = X.mean(axis=0)
X_centered = X - X_mean

# PCA directions
U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)
v1, v2 = Vh[:2]  # shape: [d]

# Project both trajectories
coords_ngd = np.stack([(X_ngd - X_mean) @ v1, (X_ngd - X_mean) @ v2], axis=1)
coords_gd  = np.stack([(X_gd  - X_mean) @ v1, (X_gd  - X_mean) @ v2], axis=1)

plt.figure(figsize=(8, 6))
plt.plot(coords_gd[:, 0], coords_gd[:, 1], '-o', color='blue', markersize=3, label='Gradient Descent')
plt.plot(coords_ngd[:, 0], coords_ngd[:, 1], '-o', color='red', markersize=3, label='Natural Gradient Descent')
plt.xlabel('Direction 1')
plt.ylabel('Direction 2')
plt.title('Optimization Trajectories (Projected to 2D)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("trajectory_comparison.png", dpi=150)
plt.close()

