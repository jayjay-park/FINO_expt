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
    cax = ax.imshow(true1, cmap=cmap)
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
folder = "noise(0.01)_NS_smooth_partial(0.1)_lr=0.5_sample=10"  #"naturalperturb_pert=0.8_30000_lr=0.5" # "." train/naturalperturb_pert=0.8_30000_lr=0.5
dest_folder = "noise(0.01)_NS_smooth_partial(0.1)_lr=0.5_sample=10" 
expt_name = "input_outdist"
np.random.seed(42)
len_traj = 2500
type_opt = "GD"
ood = False
data_type = 'NS'
initial_guess = 'smooth'
sample_idx = 0
dim = 64


# --- PATHS ---
h5_jac = f"{folder}/inversion_history_JAC_{data_type}_400_{initial_guess}_{type_opt}.h5"
h5_mse = f"{folder}/inversion_history_MSE_{data_type}_{initial_guess}_{type_opt}.h5"
h5_devito = f"{folder}/inversion_history_Devito_{data_type}_{initial_guess}_{type_opt}.h5"


# --- LOAD TRAJECTORIES ---
with h5py.File(h5_jac, "r") as f:
    path_ngd = f["a"][:].squeeze()[sample_idx]
print("done with 400")
with h5py.File(h5_mse, "r") as f:
    path_gd = f["a"][:].squeeze()[sample_idx]
print("done with MSE")
with h5py.File(h5_devito, "r") as f:
    path_d = f["a"][:].squeeze()[sample_idx]
print("done with NS")

# --- LOAD GROUND TRUTH ---
if ood == True:
    with h5py.File(f"{folder}/grf_sample_data_0.h5", "r") as f:
        x_true = torch.tensor(f["x"][:])
        i = torch.tensor(f["i"][:]).long()
        j = torch.tensor(f["j"][:]).long()
        ood_x = torch.tensor(f['ood_x'][:])
else:
    with h5py.File(f"{folder}/grf_sample_data_{data_type}_{initial_guess}_{sample_idx}.h5", "r") as f:
        x_true = torch.tensor(f["x"][:])
        i = torch.tensor(f["i"][:]).long()
        j = torch.tensor(f["j"][:]).long()

# plot_single(x_true.squeeze(), f"{dest_folder}/grf_sample_0.png", "viridis")
if ood == True:
    plot_single(ood_x.squeeze(), f"{dest_folder}/grf_ood_0.png", "viridis")

# --- FLATTEN ---
# X_ngd_50 = path_ngd_50.squeeze(0).reshape(path_ngd_50.shape[1], -1)[:len_traj]
# X_ngd_200 = path_ngd_200.squeeze(0).reshape(path_ngd_200.shape[1], -1)[:len_traj]

X_ngd = path_ngd.squeeze().reshape(path_ngd.shape[0], -1)[:len_traj]
X_gd = path_gd.squeeze().reshape(path_gd.shape[0], -1)[:len_traj]
X_d = path_d.squeeze().reshape(path_d.shape[0], -1)[:len_traj]

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
    x = torch.tensor(x_flat.reshape(x_true.shape), dtype=torch.float32)
    if ood == True:
        return mse(x, ood_x.squeeze()).item()
    else:
        return mse(x, x_true).item()

# --- 2D Grid for Loss Evaluation ---
# all_coords = np.vstack([coords_d, coords_ngd, coords_gd, coords_ngd_50, coords_ngd_200])
all_coords = np.vstack([coords_ngd, coords_d, coords_gd])
max_extent = np.max(np.abs(all_coords)) * 1.1
xx, yy = np.meshgrid(np.linspace(-max_extent, max_extent, n_grid),
                     np.linspace(-max_extent, max_extent, n_grid))
X_grid = X_mean[None, :] + xx[..., None]*v1 + yy[..., None]*v2
print("v1 shape", v1.shape, X_grid.shape)
X_grid_flat = X_grid.reshape(-1, v1.shape[0])

print("Evaluating loss...")
loss_vals = np.array([loss_fn(x) for x in X_grid_flat]).reshape(n_grid, n_grid)

# --- PLOT ---
plt.figure(figsize=(8, 6))
contour = plt.contour(xx, yy, loss_vals, levels=30, cmap='jet')
plt.clabel(contour, fmt="%.4e", inline=True, fontsize=10)
# log_loss_vals = np.log10(loss_vals)
# contour = plt.contour(xx, yy, log_loss_vals, levels=20, cmap='viridis')
# plt.clabel(contour, fmt=lambda x: f"1e{x:.1f}", inline=True, fontsize=8)
# plt.colorbar(contour, label="log10(Loss Value)")


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
plot_traj(coords_ngd, cm.Blues, r"FINO ($r=400$)")
# plot_traj(coords_ngd_200, cm.Purples, r"FINO ($r=200$)")
# plot_traj(coords_ngd_50, cm.Greys, r"FINO ($r=50$)")
plot_traj(coords_gd, cm.Reds, "MSE-FNO")

plt.xlabel("Principal Direction 1", fontsize=14)
plt.ylabel("Principal Direction 2", fontsize=14)
plt.title("Loss Contours & Optimization Trajectories", fontsize=16, fontweight='bold')
plt.grid(True)
plt.legend(loc="upper right", fontsize=14)
plt.tight_layout()
plt.savefig(f"{dest_folder}/loss_landscape_with_markers_{expt_name}_{len_traj}_{type_opt}.png", dpi=150)
plt.close()


# --- Save final model iterates and x_true as images ---
if ood == True:
    final_states = {
        "Numerical Simulator": X_d[-1],
        "Jvp (FIM: 50)": X_ngd_50[-1],
        "Jvp (FIM: 200)": X_ngd_200[-1],
        "Jvp (FIM: 400)": X_ngd[-1],
        "MSE": X_gd[-1],
        r"OOD $\mathbf{a}$": ood_x.squeeze().numpy().reshape(-1)
    }
else:
    final_states = {
        "Numerical Simulator": X_d[-1],
        # "Jvp (FIM: 50)": X_ngd_50[-1],
        # "Jvp (FIM: 200)": X_ngd_200[-1],
        "Jvp (FIM: 400)": X_ngd[-1],
        "MSE": X_gd[-1],
        r"Ground Truth $\mathbf{a}$": x_true.squeeze().numpy().reshape(-1)
    }

# Determine shape (assume square if unknown)
n_pixels = final_states["MSE"].shape[0]
side = int(np.sqrt(n_pixels))

plt.figure(figsize=(10, 2))
for i, (label, arr_flat) in enumerate(final_states.items()):
    arr_2d = arr_flat.reshape(side, side)
    plt.subplot(1, len(final_states), i + 1)
    im = plt.imshow(arr_2d, cmap="viridis")
    plt.title(label, fontsize=8)
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f"{dest_folder}/final_model_iterates_vs_ground_truth_{expt_name}_{len_traj}_{type_opt}.png")
plt.close()




# ===== Overlay "gradient fans" at each NS iterate =====
import h5py
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# --- helpers ---
def load_gradients(h5_path, key="g", T=None, sample_idx=0):
    with h5py.File(h5_path, "r") as f:
        G = f[key][:]  # (samples, iters, H, W)
    s, t, h, w = G.shape
    if T is None: T = t
    return G.reshape(s, t, h*w)[sample_idx, :T]   # (T, d)

def unit_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M / n

def angle_deg(Gm, Gns):
    dot = np.sum(Gm * Gns, axis=1)
    n1  = np.linalg.norm(Gm,  axis=1) + 1e-12
    n2  = np.linalg.norm(Gns, axis=1) + 1e-12
    c   = np.clip(dot/(n1*n2), -1.0, 1.0)
    return np.degrees(np.arccos(c))  # (T,)

def rel_l2(Gm, Gns):
    n2  = np.linalg.norm(Gns, axis=1) + 1e-12
    return np.linalg.norm(Gm - Gns, axis=1) / n2  # (T,)

def project_vec2(v):  # v: (d,)
    return np.array([v @ v1, v @ v2])

def project_rows(M):  # M: (T,d) -> (T,2)
    return np.stack([M @ v1, M @ v2], axis=1)

# --- load gradients (w.r.t NS) ---
grad_ns_file = f"{folder}/inversion_history_gradient_NS_Devito_{initial_guess}_{type_opt}.h5"
grad_jac50   = f"{folder}/inversion_history_gradient_NS_JAC_50_{initial_guess}_{type_opt}.h5"
grad_jac200  = f"{folder}/inversion_history_gradient_NS_JAC_200_{initial_guess}_{type_opt}.h5"
grad_jac400  = f"{folder}/inversion_history_gradient_NS_JAC_400_{initial_guess}_{type_opt}.h5"
grad_mse     = f"{folder}/inversion_history_gradient_NS_MSE_{initial_guess}_{type_opt}.h5"

T = len_traj  # number of iterates you plotted
G_ns  = load_gradients(grad_ns_file,  T=T)
G_50  = load_gradients(grad_jac50,    T=T)
G_200 = load_gradients(grad_jac200,   T=T)
G_400 = load_gradients(grad_jac400,   T=T)
G_mse = load_gradients(grad_mse,      T=T)

# --- package models (label -> dict of arrays) ---
models = {
    r"FINO ($r=50$)"  : {"G": G_50,  "outline": "#7f7f7f"},  # grey
    r"FINO ($r=200$)" : {"G": G_200, "outline": "#9467bd"},  # purple
    r"FINO ($r=400$)" : {"G": G_400, "outline": "#1f77b4"},  # blue
    "MSE-FNO"         : {"G": G_mse, "outline": "#d62728"},  # red
}

# --- compute per-model θ and magnitude arrays ---
for name, dct in models.items():
    Gm = dct["G"]
    dct["theta_deg"] = angle_deg(Gm, G_ns)            # color
    dct["mag_rel2"]  = rel_l2(Gm,  G_ns)              # length scale
    dct["u_proj"]    = project_rows(unit_rows(Gm))    # direction in PCA plane (T,2)

# --- robust length scaling shared across models ---
all_mags = np.concatenate([m["mag_rel2"] for m in models.values()])
p95 = float(np.nanpercentile(all_mags, 95))
for dct in models.values():
    dct["len_scale"] = np.clip(dct["mag_rel2"] / (p95 + 1e-12), 0, 1)  # in [0,1]
    # final arrow vector in PCA plane
    dct["arrow"] = dct["u_proj"] * dct["len_scale"][:, None]

# --- color by θ (common colormap across all models) ---
all_theta = np.concatenate([m["theta_deg"] for m in models.values()])
theta_max = max(1.0, float(np.nanmax(all_theta)))
norm = Normalize(vmin=0, vmax=theta_max)
cmap = plt.get_cmap("viridis")

# --- draw fans at each NS iterate (subsample for clarity) ---
ax = plt.gca()  # current axes with contours/trajectories
step = every_n  # e.g., draw fans every 'every_n' iters
for t in range(0, T-1, step):
    x0, y0 = coords_d[t]  # anchor at NS point
    for name, dct in models.items():
        # tiny radial jitter to prevent exact overlap of fan bases (optional)
        jitter = 0.0
        x_base = x0 + jitter * 0.0
        y_base = y0 + jitter * 0.0

        dx, dy = dct["arrow"][t]            # arrow vector in PCA plane
        color  = cmap(norm(dct["theta_deg"][t]))

        # Each model gets its own outline color so you can tell models apart
        ax.quiver(
            x_base, y_base, dx, dy,
            angles='xy', scale_units='xy', scale=1.0,
            color=color, edgecolor=dct["outline"],
            linewidth=0.5, width=0.004,
            headwidth=3.0, headlength=5.0, headaxislength=4.5,
            alpha=0.95, zorder=5
        )

# --- colorbar for angle, legend for model outlines ---
sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.015)
cbar.set_label(r'Angular deviation $\theta$ (deg)')

from matplotlib.lines import Line2D
legend_handles = [Line2D([0],[0], color=m["outline"], lw=3, label=lbl)
                  for lbl, m in models.items()]
leg1 = ax.legend(handles=legend_handles, title="Model (arrow outline)", loc="lower left", fontsize=10)
ax.add_artist(leg1)

# optional: gentle padding
xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin - 0.02*(ymax-ymin), ymax)

# Save another version with fans
plt.tight_layout()
plt.savefig(f"{dest_folder}/loss_landscape_with_fans_{expt_name}_{len_traj}_{type_opt}.png", dpi=150)
print("Saved:", f"{dest_folder}/loss_landscape_with_fans_{expt_name}_{len_traj}_{type_opt}.png")
