# import numpy as np
# import matplotlib.pyplot as plt
# import h5py
# import torch
# from matplotlib import cm
# from models.ns_inversion import NSModel  # Your model
# import os
# from groundwater.devito_op import GroundwaterModel, GroundwaterLayer
# from utils import get_dataset, load_config, get_model  # Your utils

# # === CONFIGURATION ===
# device = "cuda" if torch.cuda.is_available() else "cpu"
# initial_guess = "smooth"

# JAC_config = load_config("configs/eigenvectors/e_400.yaml")
# JAC_ckpt_path = f"checkpoints/n=400_e=400_m=FNO_s=RFS_l=JAC_20250617_131205/n=400_e=400_m=FNO_s=RFS_l=JAC_epoch=299_val_rel_l2_loss=0.0156.ckpt"

# MSE_config = load_config("configs/darcy_MSE.yaml")
# MSE_ckpt_path = "checkpoints/DARCY_MSE/Darcy_training_epoch=249_val_rel_l2_loss=0.0009_MSE_May14.ckpt"
    

# MSE_model = NSModel.load_from_checkpoint(MSE_ckpt_path).eval().to(device).eval()
# JAC_model = NSModel.load_from_checkpoint(JAC_ckpt_path).eval().to(device).eval()
# forcing_term = torch.zeros(128, 128)
# groundwater_model = GroundwaterModel(forcing_term.shape[0])
# model = lambda x: groundwater_model(x, forcing_term)

# folder = "."
# n_grid = 100

# # === PATHS ===
# h5_jac_200 = f"{folder}/inversion_history_JAC_400_{initial_guess}.h5"
# h5_mse = f"{folder}/inversion_history_MSE_{initial_guess}.h5"
# h5_devito = f"{folder}/inversion_history_Devito_{initial_guess}.h5"

# # === LOAD TRAJECTORIES ===
# with h5py.File(h5_jac_200, "r") as f:
#     path_ngd = f["a"][:]
# with h5py.File(h5_mse, "r") as f:
#     path_gd = f["a"][:]
# with h5py.File(h5_devito, "r") as f:
#     path_d = f["a"][:]

# # === LOAD TRUE DATA ===
# with h5py.File("grf_sample_data_0.h5", "r") as f:
#     x_true = torch.tensor(f["x"][:], dtype=torch.float32).to(device)
#     y_true = torch.tensor(f["y"][:], dtype=torch.float32).to(device)
#     i = torch.tensor(f["i"][:], dtype=torch.long).to(device)
#     j = torch.tensor(f["j"][:], dtype=torch.long).to(device)

# # === FLATTEN AND SUBSAMPLE ===
# X_ngd = path_ngd.squeeze(0).reshape(path_ngd.shape[1], -1)[::20]
# X_gd = path_gd.squeeze(0).reshape(path_gd.shape[1], -1)[::20]
# X_d = path_d.squeeze(0).reshape(path_d.shape[1], -1)[::20]

# # === USE NGD + Devito FOR PCA BASIS ===
# X_basis = np.vstack([X_d, X_ngd])
# X_mean = X_basis.mean(axis=0)
# X_centered = X_basis - X_mean
# U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)
# v1, v2 = Vh[:2]
# H, W = x_true.shape[-2:]  # assume shape (1,1,H,W)

# # === DEFINE LOSS FUNCTION USING MODEL ===
# mse = torch.nn.MSELoss()
# def loss_fn(x_flat):
#     x = x_flat.reshape(1, 1, H, W)
#     x = torch.tensor(x, dtype=torch.float32).to(device)
#     y = y_true.squeeze()
#     with torch.no_grad():
#         pred = model(x)
#         print(y.shape, pred.shape)
#         return mse(pred[i, j], y[i, j]).item()

# # === PROJECT TRAJECTORIES TO 2D ===
# def project(x): return np.array([(x - X_mean) @ v1, (x - X_mean) @ v2])
# coords_d   = np.array([project(x) for x in X_d])
# coords_ngd = np.array([project(x) for x in X_ngd])
# coords_gd  = np.array([project(x) for x in X_gd])

# # === GENERATE 2D GRID ===
# all_coords = np.vstack([coords_d, coords_ngd, coords_gd])
# max_extent = np.max(np.abs(all_coords)) * 1.2
# xx, yy = np.meshgrid(
#     np.linspace(-max_extent, max_extent, n_grid),
#     np.linspace(-max_extent, max_extent, n_grid)
# )
# X_grid_full = X_mean[None, :] + xx[..., None]*v1 + yy[..., None]*v2
# X_grid_flat = X_grid_full.reshape(-1, v1.shape[0])

# # === EVALUATE LOSS ON GRID ===
# print("Evaluating loss over 2D grid...")
# loss_vals = np.array([loss_fn(x) for x in X_grid_flat])
# loss_vals = loss_vals.reshape(n_grid, n_grid)

# # === PLOT ===
# plt.figure(figsize=(9, 7))
# contour = plt.contourf(xx, yy, loss_vals, levels=50, cmap='viridis')

# plt.plot(coords_gd[:, 0], coords_gd[:, 1], 'o-', color='red', label='MSE')
# plt.plot(coords_ngd[:, 0], coords_ngd[:, 1], 'o-', color='blue', label='Jvp (FIM: 400)')
# plt.plot(coords_d[:, 0], coords_d[:, 1], 'o-', color='green', label='Numerical Simulator')

# # Mark start and end
# plt.plot(coords_d[0, 0], coords_d[0, 1], 'o', color='darkgreen', label='Start', markersize=6)
# plt.plot(coords_d[-1, 0], coords_d[-1, 1], 'X', color='black', label='End', markersize=6)

# plt.xlabel("Direction 1")
# plt.ylabel("Direction 2")
# plt.title("Loss Landscape and Optimization Trajectories (2D Plane)")
# plt.colorbar(contour, label="Loss Value")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("loss_landscape_model_based.png", dpi=150)
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from matplotlib import cm

# --- CONFIGURATION ---
initial_guess = "smooth"
device = "cpu"
n_grid = 100
every_n = 20  # mark every n-th point
folder = "."

# --- PATHS ---
h5_jac = f"{folder}/inversion_history_JAC_400_{initial_guess}.h5"
h5_mse = f"{folder}/inversion_history_MSE_{initial_guess}.h5"
h5_devito = f"{folder}/inversion_history_Devito_{initial_guess}.h5"

# --- LOAD TRAJECTORIES ---
with h5py.File(h5_jac, "r") as f:
    path_ngd = f["a"][:]
with h5py.File(h5_mse, "r") as f:
    path_gd = f["a"][:]
with h5py.File(h5_devito, "r") as f:
    path_d = f["a"][:]

# --- LOAD GROUND TRUTH ---
with h5py.File(f"{folder}/grf_sample_data_0.h5", "r") as f:
    x_true = torch.tensor(f["x"][:])
    i = torch.tensor(f["i"][:]).long()
    j = torch.tensor(f["j"][:]).long()

# --- FLATTEN ---
X_ngd = path_ngd.squeeze(0).reshape(path_ngd.shape[1], -1)[:100]
print("ngd", X_ngd.shape)
X_gd = path_gd.squeeze(0).reshape(path_gd.shape[1], -1)[:100]
X_d = path_d.squeeze(0).reshape(path_d.shape[1], -1)[:100]

# --- PCA Projection Basis (v1, v2) ---
X_basis = np.vstack([X_d, X_ngd, X_gd])
X_mean = X_basis.mean(axis=0)
U, S, Vh = np.linalg.svd(X_basis - X_mean, full_matrices=False)
v1, v2 = Vh[:2]

def project(x): return np.array([(x - X_mean) @ v1, (x - X_mean) @ v2])
coords_d = np.array([project(x) for x in X_d])
coords_ngd = np.array([project(x) for x in X_ngd])
coords_gd = np.array([project(x) for x in X_gd])

# --- Define loss ---
mse = torch.nn.MSELoss()
def loss_fn(x_flat):
    x = torch.tensor(x_flat.reshape(x_true.shape), dtype=torch.float32)
    return mse(x, x_true).item()

# --- 2D Grid for Loss Evaluation ---
all_coords = np.vstack([coords_d, coords_ngd, coords_gd])
max_extent = np.max(np.abs(all_coords)) * 1.1
xx, yy = np.meshgrid(np.linspace(-max_extent, max_extent, n_grid),
                     np.linspace(-max_extent, max_extent, n_grid))
X_grid = X_mean[None, :] + xx[..., None]*v1 + yy[..., None]*v2
X_grid_flat = X_grid.reshape(-1, v1.shape[0])

print("Evaluating loss...")
loss_vals = np.array([loss_fn(x) for x in X_grid_flat]).reshape(n_grid, n_grid)

# --- PLOT ---
plt.figure(figsize=(10, 8))
contour = plt.contour(xx, yy, loss_vals, levels=30, cmap='viridis')
plt.clabel(contour, fmt="%.2e", inline=True, fontsize=8)

# Plot color-varying trajectories
def plot_traj(coords, cmap, label):
    T = len(coords)
    for t in range(T - 1):
        plt.plot(*zip(coords[t], coords[t+1]), color=cmap(t / T))
    dots = coords[::every_n]
    plt.plot(dots[:, 0], dots[:, 1], 'o', color=cmap(0.7), markersize=3, label=f"{label} (every {every_n})")
    if label == "Numerical Simulator":
        plt.plot(coords[0, 0], coords[0, 1], 'o', color='black', markersize=6, label=f"Start")
        plt.plot(coords[-1, 0], coords[-1, 1], 'X', color='black', markersize=6, label=f"End")
    else:
        plt.plot(coords[0, 0], coords[0, 1], 'o', color='black', markersize=6)
        plt.plot(coords[-1, 0], coords[-1, 1], 'X', color='black', markersize=6)

plot_traj(coords_d, cm.Greens, "Numerical Simulator")
plot_traj(coords_ngd, cm.Blues, "Jvp (FIM: 400)")
plot_traj(coords_gd, cm.Reds, "MSE")

plt.xlabel("Direction 1")
plt.ylabel("Direction 2")
plt.title("Loss Contours and Optimization Trajectories")
plt.colorbar(contour, label="Loss Value")
plt.grid(True)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("loss_landscape_with_markers.png", dpi=150)
plt.show()
