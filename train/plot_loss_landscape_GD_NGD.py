import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from matplotlib import cm
from matplotlib import colors

def plot_single(true1, path, cmap="jet", vmin=None, vmax=None):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})
    print("vmin", vmin, vmax)
    if vmin != 0:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if (vmin is not None and vmax is not None) else colors.CenteredNorm()
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax) if (vmin is not None and vmax is not None) else colors.CenteredNorm()
    
    fig, ax = plt.subplots()
    cax = ax.imshow(true1, cmap=cmap, norm=norm)
    plt.colorbar(cax, ax=ax, fraction=0.045, pad=0.06)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

# --- CONFIGURATION ---
initial_guess = "prior_mean"
device = "cpu"
n_grid = 50 #100
every_n = 10  # mark every n-th point
folder = "." #"naturalperturb_pert=0.8_30000_lr=0.5" # "." train/naturalperturb_pert=0.8_30000_lr=0.5
folder_NGD = "prior_mean_wonoise_NGD"
folder_GD = "prior_mean_wonoise"
dest_folder = "prior_mean_wonoise"
expt_name = "input_ood=0_noise=0."
np.random.seed(42)
len_traj = 999
type_opt = "GD"
ood = False

# --- PATHS ---
# h5_jac_50 = f"{folder}/inversion_history_JAC_50_{initial_guess}_{type_opt}.h5"
# h5_jac_200 = f"{folder}/inversion_history_JAC_200_{initial_guess}_{type_opt}.h5"
h5_jac = f"{folder_GD}/inversion_history_JAC_400_{initial_guess}.h5"
h5_mse = f"{folder_GD}/inversion_history_MSE_{initial_guess}.h5"
h5_devito = f"{folder_NGD}/inversion_history_Devito_{initial_guess}_NGD.h5"


# --- LOAD TRAJECTORIES ---
# with h5py.File(h5_jac_50, "r") as f:
#     path_ngd_50 = f["a"][:]
# print("done with 50")
# with h5py.File(h5_jac_200, "r") as f:
#     path_ngd_200 = f["a"][:]
# print("done with 200")
with h5py.File(h5_jac, "r") as f:
    path_ngd = f["a"][:]
print("done with 400")
with h5py.File(h5_mse, "r") as f:
    path_gd = f["a"][:]
print("done with MSE")
with h5py.File(h5_devito, "r") as f:
    path_d = f["a"][:]
print("done with NS")

# --- LOAD GROUND TRUTH ---
# with h5py.File(f"{folder}/grf_sample_data_0.h5", "r") as f:
#     x_true = torch.tensor(f["x"][:])
#     i = torch.tensor(f["i"][:]).long()
#     j = torch.tensor(f["j"][:]).long()
#     ood_x = torch.tensor(f['ood_x'][:])
# plot_single(x_true.squeeze(), f"{dest_folder}/grf_sample_0.png", "viridis")
# plot_single(ood_x.squeeze(), f"{dest_folder}/grf_ood_0.png", "viridis")

# --- FLATTEN ---
# X_ngd_50 = path_ngd_50.squeeze(0).reshape(path_ngd_50.shape[1], -1)[:len_traj]
# X_ngd_200 = path_ngd_200.squeeze(0).reshape(path_ngd_200.shape[1], -1)[:len_traj]
X_ngd = path_ngd.squeeze(0).reshape(path_ngd.shape[1], -1)[:len_traj]
X_gd = path_gd.squeeze(0).reshape(path_gd.shape[1], -1)[:len_traj]
X_d = path_d.squeeze(0).reshape(path_d.shape[1], -1)[:len_traj]
print("X_d", X_d[-1].shape)

# --- PCA Projection Basis (v1, v2) ---
# X_basis = np.vstack([X_d, X_ngd_200, X_ngd, X_gd]) # X_d
X_basis = np.vstack([X_d])
X_mean = X_basis.mean(axis=0)
U, S, Vh = np.linalg.svd(X_basis - X_mean, full_matrices=False)
v1, v2 = Vh[:2]

print("after pca")

# # Get dimensionality
# dim = X_d.shape[1]

# # Sample two random orthonormal directions
# random_matrix = np.random.randn(2, dim)
# q, _ = np.linalg.qr(random_matrix.T)  # Orthonormalize
# v1, v2 = q.T  # v1, v2 are now orthonormal directions in R^dim


def project(x): return np.array([(x - X_mean) @ v1, (x - X_mean) @ v2])
coords_d = np.array([project(x) for x in X_d])
coords_ngd = np.array([project(x) for x in X_ngd])
# coords_ngd_200 = np.array([project(x) for x in X_ngd_200])
# coords_ngd_50 = np.array([project(x) for x in X_ngd_50])
coords_gd = np.array([project(x) for x in X_gd])

# --- Define loss ---
mse = torch.nn.MSELoss()
def loss_fn(x_flat):
    x = torch.tensor(x_flat.reshape(128, 128), dtype=torch.float32)
    X_d_final = torch.tensor(X_d[-1]).reshape(128,128)
    if ood == True:
        return mse(x, ood_x).item()
    else:
        return mse(x, X_d_final)

# --- 2D Grid for Loss Evaluation ---
# all_coords = np.vstack([coords_d, coords_ngd, coords_gd, coords_ngd_50, coords_ngd_200])
all_coords = np.vstack([coords_ngd, coords_d, coords_gd])
max_extent = np.max(np.abs(all_coords)) * 1.1
xx, yy = np.meshgrid(np.linspace(-max_extent, max_extent, n_grid),
                     np.linspace(-max_extent, max_extent, n_grid))
X_grid = X_mean[None, :] + xx[..., None]*v1 + yy[..., None]*v2
X_grid_flat = X_grid.reshape(-1, v1.shape[0])

print("Evaluating loss...")
loss_vals = np.array([loss_fn(x) for x in X_grid_flat]).reshape(n_grid, n_grid)

# --- PLOT ---
plt.figure(figsize=(10, 8))
contour = plt.contour(xx, yy, loss_vals, levels=30, cmap='jet')
plt.clabel(contour, fmt="%.4e", inline=True, fontsize=8)


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

plot_traj(coords_d, cm.Greens, "Numerical Simulator with NGD")
plot_traj(coords_ngd, cm.Blues, "Jvp (FIM: 400) with GD")
# plot_traj(coords_ngd_200, cm.Purples, "Jvp (FIM: 200)")
# plot_traj(coords_ngd_50, cm.Greys, "Jvp (FIM: 50)")
plot_traj(coords_gd, cm.Reds, "MSE with GD")

plt.xlabel("Principal Direction 1")
plt.ylabel("Principal Direction 2")
plt.title("Loss Contours and Optimization Trajectories")
plt.grid(True)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{dest_folder}/loss_landscape_with_markers_{expt_name}_{len_traj}_{type_opt}_GD_NGD.png", dpi=150)

