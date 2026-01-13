import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from matplotlib import cm, colors
from typing import Dict, Callable, List, Tuple

# ------------------------
# Config
# ------------------------
device = "cpu"
n_grid = 50
every_n = 10
len_traj = 999
save_path = "prior_mean_wonoise/loss_landscape_with_markers.png"

# H5 paths (example: your original three)
paths = {
    "NGD (Jvp FIM:400) + GD": "prior_mean_wonoise/inversion_history_JAC_400_prior_mean.h5",
    "MSE + GD":               "prior_mean_wonoise/inversion_history_MSE_prior_mean.h5",
    "Numerical Simulator (NGD)": "prior_mean_wonoise_NGD/inversion_history_Devito_prior_mean_NGD.h5",
}

# Choose which trajectories define the PCA plane (often: include all)
basis_labels_for_pca = ["NGD (Jvp FIM:400) + GD", "MSE + GD", "Numerical Simulator (NGD)"]

# Loss selection:
# - "sim_target": MSE to a reference field (e.g., final simulator iterate)
# - "network":    call a provided net loss closure per grid point
loss_mode = "sim_target"

# If loss_mode = "sim_target", choose your reference:
reference_from = "Numerical Simulator (NGD)"  # use final iterate of this traj as target
target_shape = (128, 128)

# If loss_mode = "network", provide a callable taking a (128,128) float32 torch.Tensor -> scalar float
def net_loss_fn(x_tensor: torch.Tensor) -> float:
    """
    Example stub for trained-network loss.
    Replace the body with your actual model/physics residual computation.
    x_tensor shape: [128,128], dtype float32, on CPU here for simplicity.
    Must return a Python float.
    """
    # Example: pretend the network returns a prediction g(x), and we compare to y_obs
    # with mse. Replace with your real code.
    # y_pred = trained_model(x_tensor.unsqueeze(0).unsqueeze(0))  # [1,1,128,128] for example
    # loss = torch.nn.functional.mse_loss(y_pred.squeeze(), y_obs_tensor)
    # return float(loss.item())
    raise NotImplementedError("Plug your trained network loss here.")

# ------------------------
# Helpers
# ------------------------
def load_traj(h5_path: str, key: str = "a", take: int = None) -> np.ndarray:
    """
    Returns flattened trajectory array of shape [T, D].
    Assumes stored as [1, T, H, W] or [1, T, ...] and squeezes dim 0.
    """
    with h5py.File(h5_path, "r") as f:
        arr = f[key][:]
    arr = arr.squeeze(0)  # [T, ...]
    T = arr.shape[0] if take is None else min(take, arr.shape[0])
    flat = arr[:T].reshape(T, -1)
    return flat

def get_all_trajs(paths_dict: Dict[str, str], take: int) -> Dict[str, np.ndarray]:
    out = {}
    for label, p in paths_dict.items():
        out[label] = load_traj(p, take=take)
    return out

def compute_pca_basis(X_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_basis = np.vstack(X_list)
    X_mean = X_basis.mean(axis=0)
    U, S, Vh = np.linalg.svd(X_basis - X_mean, full_matrices=False)
    v1, v2 = Vh[:2]
    return X_mean, v1, v2

def project_coords(X: np.ndarray, X_mean: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    # Returns [T, 2]
    delta = X - X_mean
    return np.stack([delta @ v1, delta @ v2], axis=1)

def make_grid(X_mean: np.ndarray, v1: np.ndarray, v2: np.ndarray, coords_list: List[np.ndarray], n_grid: int):
    all_coords = np.vstack(coords_list)
    max_extent = float(np.max(np.abs(all_coords)) * 1.1 + 1e-8)
    xx, yy = np.meshgrid(np.linspace(-max_extent, max_extent, n_grid),
                         np.linspace(-max_extent, max_extent, n_grid))
    X_grid = X_mean[None, :] + xx[..., None] * v1 + yy[..., None] * v2
    return xx, yy, X_grid

def mse_to_target(x_flat: np.ndarray, target_flat: np.ndarray, shp: Tuple[int, int]) -> float:
    x = torch.tensor(x_flat.reshape(shp), dtype=torch.float32)
    tgt = torch.tensor(target_flat.reshape(shp), dtype=torch.float32)
    return float(torch.nn.functional.mse_loss(x, tgt).item())

# ------------------------
# Load trajectories
# ------------------------
trajs = get_all_trajs(paths, take=len_traj)

# Choose target for "sim_target" loss
if loss_mode == "sim_target":
    assert reference_from in trajs, f"reference_from='{reference_from}' not found in trajs"
    target_flat = trajs[reference_from][-1].copy()  # final iterate of chosen trajectory

# PCA plane
X_mean, v1, v2 = compute_pca_basis([trajs[lbl] for lbl in basis_labels_for_pca])

# Project each trajectory
coords = {lbl: project_coords(X, X_mean, v1, v2) for lbl, X in trajs.items()}

# Grid for loss evaluation
xx, yy, X_grid = make_grid(X_mean, v1, v2, list(coords.values()), n_grid=n_grid)
X_grid_flat = X_grid.reshape(-1, X_mean.shape[0])

# ------------------------
# Loss evaluation on grid
# ------------------------
print("Evaluating loss grid...")
loss_vals = np.zeros((n_grid * n_grid,), dtype=np.float64)

if loss_mode == "sim_target":
    # Vectorized batching for speed
    B = 2048  # batch size for loss eval
    for s in range(0, X_grid_flat.shape[0], B):
        e = min(s + B, X_grid_flat.shape[0])
        batch = X_grid_flat[s:e]
        # Pure NumPy->Torch MSE to reference
        losses = [mse_to_target(b, target_flat, target_shape) for b in batch]
        loss_vals[s:e] = np.asarray(losses, dtype=np.float64)

elif loss_mode == "network":
    # Use your trained network loss callable
    B = 256
    torch.set_grad_enabled(False)
    for s in range(0, X_grid_flat.shape[0], B):
        e = min(s + B, X_grid_flat.shape[0])
        batch = X_grid_flat[s:e]
        losses = []
        for b in batch:
            x_t = torch.tensor(b.reshape(target_shape), dtype=torch.float32, device=device)
            losses.append(float(net_loss_fn(x_t)))
        loss_vals[s:e] = np.asarray(losses, dtype=np.float64)
    torch.set_grad_enabled(True)
else:
    raise ValueError(f"Unknown loss_mode: {loss_mode}")

loss_vals = loss_vals.reshape(n_grid, n_grid)

# ------------------------
# Plot
# ------------------------
plt.figure(figsize=(10, 8))
contour = plt.contour(xx, yy, loss_vals, levels=30, cmap="jet")
plt.clabel(contour, fmt="%.3e", inline=True, fontsize=8)

def plot_traj(c: np.ndarray, cmap, label: str, mark_start_end: bool = True):
    T = len(c)
    for t in range(T - 1):
        plt.plot(*zip(c[t], c[t+1]), color=cmap(t / T))
    dots = c[::every_n]
    plt.plot(dots[:, 0], dots[:, 1], 'o', color=cmap(0.7), markersize=3, label=f"{label} (every {every_n})")
    if mark_start_end:
        plt.plot(c[0, 0], c[0, 1], 'o', color='black', markersize=6, label="Start")
        plt.plot(c[-1, 0], c[-1, 1], 'X', color='black', markersize=6, label="End")

# Preserve your color choices
for lbl, c in coords.items():
    if "Numerical" in lbl:
        plot_traj(c, cm.Greens, lbl)
    elif "MSE" in lbl:
        plot_traj(c, cm.Reds, lbl)
    else:
        plot_traj(c, cm.Blues, lbl)

plt.xlabel("Principal Direction 1")
plt.ylabel("Principal Direction 2")
title_loss = "MSE to simulator final" if loss_mode == "sim_target" else "Trained-network loss"
plt.title(f"Loss Contours ({title_loss}) and Optimization Trajectories")
plt.grid(True)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(save_path, dpi=150)
plt.close()
print(f"Saved figure to: {save_path}")

