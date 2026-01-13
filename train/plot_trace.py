# plot trace
import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys, h5py
from utils import *
from utils_plot import *
from utils_inversion import *
from models.ns_inversion import NSModel

# ---- Load trained model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_config("configs/VMB_MSE.yaml")

# 150 x 150
# ckpt_path = "checkpoints/VMB_training_20251111_033550/VMB_training_epoch=030_val_rel_l2_loss=0.9222.ckpt"
# ckpt_path = "checkpoints/VMB_training_20251111_033550/VMB_training_epoch=008_val_rel_l2_loss=0.8566.ckpt"
# ckpt_path = "checkpoints/VMB_training_20251111_033550/VMB_training_epoch=185_val_rel_l2_loss=0.9859.ckpt"
ckpt_path = "checkpoints/VMB_training_20251111_033550/VMB_training_epoch=110_val_rel_l2_loss=0.9879.ckpt"

# 80 x 80
# ckpt_path = "checkpoints/VMB_training_20251111_032041/VMB_training_epoch=040_val_rel_l2_loss=0.9582.ckpt"

model = NSModel.load_from_checkpoint(ckpt_path).eval().to(device)

# ---- Make output folder ----
outdir = "plot/inversion_VMB"
os.makedirs(outdir, exist_ok=True)

# ---- Dataset ----
num_sample = 1
config.data_settings["batch_size"] = 1
dataset = get_dataset(config.experiment.dataset_type, config.data_settings)
dataloader = dataset.get_dataloader(offset=1, limit=num_sample, shuffle=False)

# ---- Normalization constants ----
stats = np.load("vmb_global_max.npz")
v_max = float(stats["v_max"])
y_max = float(stats["y_max"])
print(f"Normalization constants: v_max={v_max:.3f}, y_max={y_max:.3e}")

# ---- Load one batch ----
batch = next(iter(dataloader))
x = batch["x"].squeeze().cpu().numpy()  # (nx, ny, 2)
y_true = batch["y"].squeeze().cpu().numpy()   # (nx, ny)
print("batch", batch)

# ---- Normalize inputs (same as training) ----
x_norm = x #/ v_max
x_torch = torch.from_numpy(x_norm).unsqueeze(0).to(device)  # (1, 2, nx, ny)
plot_single(x_norm[0], f'{outdir}/sanity_check_0', vmin=x_norm[0].min)
plot_single(x_norm[1], f'{outdir}/sanity_check_1', vmin=x_norm[1].min)

# ---- Predict ----
with torch.no_grad():
    y_pred_norm = model(x_torch).squeeze(0).squeeze(0).cpu().numpy()  # (nx, ny)

# ---- De-normalize back to physical amplitude ----
y_pred = y_pred_norm #* (y_max  / 500)
y_true_phys = y_true #* (y_max / 500)

# ---- Choose vertical trace ----
nx, ny = y_pred.shape
mid_col = ny // 2
trace_pred = y_pred[:, mid_col]
trace_true = y_true_phys[:, mid_col]
depth = np.arange(nx)

# ---- Plot vertical trace ----
plt.figure(figsize=(6, 8))
plt.plot(trace_true, depth, label="GroundTruth", color="k", linewidth=2, linestyle="--")
plt.plot(trace_pred, depth, label=f"Predicted", color="r", linewidth=2, alpha=0.7)
plt.gca().invert_yaxis()
plt.xlabel("Amplitude")
plt.ylabel("Depth (m)")
plt.legend()
plt.title(f"RTM Vertical Trace at x={mid_col*12.5} (m)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(outdir, f"learned_trace"), dpi=200)
plt.close()

# ---- Overlay images ----
# fig, axs = plt.subplots(1, 2, figsize=(18, 6))

# vmax_true_98 = np.percentile(np.abs(y_true_phys), 98)
# vmax_pred_98 = np.percentile(np.abs(y_pred), 98)

# im0 = axs[0].imshow(y_true_phys, cmap="gray", aspect="auto", vmin=-vmax_true_98, vmax=vmax_true_98)
# axs[0].axvline(mid_col, color="k", linestyle="--")
# axs[0].set_title("True RTM")
# fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

# im1 = axs[1].imshow(y_pred, cmap="gray", aspect="auto", vmin=-vmax_pred_98, vmax=vmax_pred_98)
# axs[1].axvline(mid_col, color="k", linestyle="--")
# axs[1].set_title("Predicted RTM")
# fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
# 
# plt.tight_layout()
# plt.savefig(os.path.join(outdir, "learned_trace_overlay.png"), dpi=200)
# plt.close()

import matplotlib.colors as colors
from matplotlib.ticker import LogFormatterSciNotation

# RTM comparison
fig, axs = plt.subplots(1, 3, figsize=(28,5))

# Dimensions
ny, nx = y_true_phys.shape
dx = dz = 12.5
extent = [0, nx*dx, ny*dz, 0]

# ------------------------------------------------------------
# 1. Groundtruth RTM
# ------------------------------------------------------------
vmax_obs = np.percentile(np.abs(y_true_phys), 98)

im0 = axs[0].imshow(
    y_true_phys, cmap="gray", aspect="auto",
    vmin=-vmax_obs, vmax=vmax_obs,
    extent=extent
)
axs[0].set_title("Groundtruth RTM")
axs[0].set_xlabel("X (m)")
axs[0].set_ylabel("Depth (m)")

fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

# ------------------------------------------------------------
# 2. Predicted RTM
# ------------------------------------------------------------
im1 = axs[1].imshow(
    y_pred, cmap="gray", aspect="auto",
    vmin=-vmax_obs, vmax=vmax_obs,
    extent=extent
)
axs[1].set_title("Predicted RTM (forward)")
axs[1].set_xlabel("X (m)")
axs[1].set_ylabel("Depth (m)")

fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

# ------------------------------------------------------------
# 3. Residual magnitude with LOG colorbar
# ------------------------------------------------------------
residual = y_true_phys - y_pred
res_mag = residual
# vmin_log = max(1e-2, np.percentile(res_mag, 2))   # avoid zeros
vmax_log = np.percentile(res_mag, 98)
# vmax_obs = np.percentile(np.abs(y_true_phys), 98)

im2 = axs[2].imshow(
    res_mag,
    cmap="RdGy",
    aspect="auto",
    vmin=-vmax_log, vmax=vmax_log,
    extent=extent
)

axs[2].set_title("Residual magnitude")
axs[2].set_xlabel("X (m)")
axs[2].set_ylabel("Depth (m)")

# Colorbar in scientific-log format
cb = fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(outdir, "rtm_compare.png"), dpi=200)
plt.close()



# plot everything
plt.figure(figsize=(24, 4))

plt.subplot(1, 3, 1)
plt.imshow(x_torch[0, 0, :, :].detach().cpu().numpy() * v_max, cmap="turbo", aspect="auto", extent=extent)
plt.title(r"$\mathbf{m}$")
plt.xlabel("X (m)")
plt.ylabel("Depth (m)")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(x_torch[0, 1, :, :].detach().cpu().numpy() * v_max, cmap="turbo", aspect="auto", extent=extent)
plt.title(r"$\mathbf{m}_0$")
plt.xlabel("X (m)")
plt.ylabel("Depth (m)")
plt.colorbar()

plt.subplot(1, 3, 3)
# y_true_phys = y_true_phys
vmax = np.percentile(np.abs(y_true_phys), 98)
im = plt.imshow(y_true_phys, cmap="gray", vmin=-vmax, vmax=vmax, aspect="auto")
plt.title("RTM")
plt.axis("off")
# plt.colorbar()

# plt.suptitle(f"GroundTruth {bundle_id} | Background: {background_tag}")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "sample.png"), dpi=200)


print(f"✅ Trace and overlay saved to {outdir}/")
