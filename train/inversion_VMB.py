#!/usr/bin/env python3
"""
Least-squares inversion for VMB using trained NSModel (no extra normalization).

This script:
  - loads one VMB sample (x=[true, background], y=RTM)
  - uses the background as initial guess
  - iteratively updates v_est to minimize  ||F(v_est, v_bg) - y_obs||² + λ_TV·TV(v_est) + λ_L2·||v_est - v_bg||²
  - saves results and figures

Assumes dataset tensors are already normalized exactly as in training.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from utils import *
from models.ns_inversion import NSModel
from utils import *
from utils_plot import *
from utils_inversion import *
from scipy.ndimage import gaussian_filter
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

# =====================================================
# Configuration
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ckpt_path = "checkpoints/VMB_training_20251111_033550/VMB_training_epoch=030_val_rel_l2_loss=0.9222.ckpt"
# ckpt_path = "checkpoints/VMB_training_20251111_033550/VMB_training_epoch=185_val_rel_l2_loss=0.9859.ckpt"
ckpt_path = "checkpoints/n=1_m=FNO_l=JAC_e=64_20260107_132311/n=1_m=FNO_l=JAC_e=64_epoch=157_val_rel_l2_loss=1.3818.ckpt"
# config = load_config("configs/VMB_MSE.yaml")
config = load_config("configs/eigenvectors/e_128_VMB.yaml")

# inversion hyperparameters
n_iter = 150 #200
lr = 0.0008 #0.015
lambda_tv = 0. #0.0001 #0.007
lambda_l2 = 0.001 #0.005 #0.0005 #0.001
lambda_l1 = 0.
outdir = "plot/inversion_VMB_H1_eig=64_sample=1"
os.makedirs(outdir, exist_ok=True)

# =====================================================
# Load model
# =====================================================
model = NSModel.load_from_checkpoint(ckpt_path).eval().to(device)
print(f"✅ Loaded model from {ckpt_path}")

# =====================================================
# Load dataset (same pipeline as training)
# =====================================================
config.data_settings["batch_size"] = 1
dataset = get_dataset(config.experiment.dataset_type, config.data_settings)
dataloader = dataset.get_dataloader(offset=0, limit=1, shuffle=False) #offset=1, 6501, 200
batch = next(iter(dataloader))

x = batch["x"].squeeze().cpu().numpy()
y_obs = batch["y"].squeeze().cpu().numpy()
print("tag", batch['background_tag'])

# Path to one of your saved samples
# file_path = "/net/slimdata/jayjaydata2/DeFINO_Richard/datasets/datasets/dataset_VMB_easy/sample_00090.h5" #90 works well 75 104
# stats = np.load("vmb_global_max_easy.npz")
# v_max = float(stats["v_max"])
# y_max = float(stats["y_max"])
# # --- Load data ---
# with h5py.File(file_path, "r") as f:
#     x = np.array(f["x"]) / 4.763645
#     y_obs = np.array(f["y_ext_13"]) / (10757.324219 / 500)
#     print("x", x.shape)
#     # optional attributes
#     bundle_id = f.attrs.get("bundle_id", "N/A")
#     background_tag = f.attrs.get("background_tag", "N/A")

# channels
v_true = x[0, :, :]
v_bg = x[1, :, :]
v_max = v_true.max()
v_min = v_true.min()

# convert to torch
v_bg_t = torch.tensor(v_bg, dtype=torch.float32, device=device)
y_obs_t = torch.tensor(y_obs, dtype=torch.float32, device=device)
v_est = v_bg_t.clone().detach().requires_grad_(True)

# if we want to smooth in background model
# sigma = 15 #5 #5  # controls how much smoothing (in pixels)
# v_bg = gaussian_filter(v_true, sigma=sigma)
# v_est = torch.tensor(v_bg, dtype=torch.float32, device=device).requires_grad_(True)

# # --- smoothing kernel for update ---
def gaussian_smooth_torch(v, sigma=5.0):
    """
    Applies Gaussian smoothing *without breaking the graph*.
    Implemented by separable convs.
    """
    if sigma <= 0:
        return v

    radius = max(1, int(3*sigma))
    x = torch.arange(-radius, radius+1, dtype=v.dtype, device=v.device)
    k = torch.exp(-(x**2)/(2*sigma**2))
    k /= k.sum()

    ky = k.view(1,1,-1,1)
    kx = k.view(1,1,1,-1)

    v = F.conv2d(v, ky, padding=(radius,0))
    v = F.conv2d(v, kx, padding=(0,radius))
    return v


# =====================================================
# Helper functions
# =====================================================
def ensure_4d(t):
    if t.dim() == 2:  # (H,W)
        return t.unsqueeze(0).unsqueeze(0)
    if t.dim() == 3:
        return t.unsqueeze(0)
    return t

def tv_loss(img):
    """Isotropic total variation loss for a 2D field (batch,1,H,W)."""
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    # crop to common region
    dy = dy[:, :, :, :-1]
    dx = dx[:, :, :-1, :]
    eps = 1e-6
    return torch.mean(torch.sqrt(dx**2 + dy**2 + eps))

def laplacian_loss_aniso(v, dz=3.0, dx=1.0, weight=1.0):
    vz = (v[:, :, 2:, 1:-1] - 2*v[:, :, 1:-1, 1:-1] + v[:, :, :-2, 1:-1]) / dz**2
    vx = (v[:, :, 1:-1, 2:] - 2*v[:, :, 1:-1, 1:-1] + v[:, :, 1:-1, :-2]) / dx**2
    lap = vz + vx
    return weight * torch.mean(lap**2)

# # =====================================================
# # Optimizer
# # =====================================================
optimizer = torch.optim.Adam([v_est], lr=lr)

losses = []
for it in range(1, n_iter + 1):
    optimizer.zero_grad()

    # Clamp physical range
    # with torch.no_grad():
        # v_est.clamp_(v_min - 0.001, v_max + 0.001)

    # Build model input: channel 0 = current estimate, channel 1 = background
    x_in = torch.stack([v_est, v_bg_t], dim=0).unsqueeze(0)  # (1,2,H,W)
    y_pred = model(x_in).squeeze(0).squeeze(0)

    data_res = y_pred - y_obs_t
    data_loss = torch.mean(data_res**2)

    tv = 0. #tv_loss(ensure_4d(v_est))
    # l2 = torch.mean((v_est - v_bg_t)**2)
    lap = laplacian_loss_aniso(v_est.unsqueeze(0).unsqueeze(0))

    loss = data_loss + lambda_tv * tv + lambda_l2 * lap
    loss.backward()

    print("before normalizing", v_est.grad.norm())
    # torch.nn.utils.clip_grad_norm_([v_est], max_norm=1.0) # normalizing the gradient

    # g = v_est.grad
    # q = torch.quantile(g.abs(), 0.999)     # threshold for top 2%
    # g.clamp_(min=-q, max=q)

    print("after normalizing", v_est.grad.norm())


    optimizer.step()


    losses.append(loss.item())
    if it % 2 == 0 or it == 1:
        v_max_phy = 4.763645
        y_max_phy = 10757.324219
        print(f"[{it:04d}/{n_iter}] loss={loss.item():.6e} data={data_loss.item():.3e}")

        v_est_phy = v_est.detach().squeeze().cpu().numpy() * v_max_phy
        v_true_phy = v_true * v_max_phy
        plot_single(v_true_phy, f"{outdir}/vel_true.png", "turbo", vmin=v_true_phy.min(), vmax=v_true_phy.max(), show_cbar=True)
        plot_single(v_est_phy, f"{outdir}/vel_iter={it}.png", "turbo", vmin=v_est_phy.min(), vmax=v_est_phy.max(), show_cbar=True)

        
        y_pred_np = y_pred.detach().squeeze().cpu().numpy()
        vmax_obs = np.percentile(np.abs(y_pred_np), 98)
        plot_single(y_pred_np, f"{outdir}/obs_iter={it}.png", "gray", vmin=-vmax_obs, vmax=vmax_obs, show_cbar=True)

        residual = (y_pred_np - y_obs)
        # residual = np.log(residual)
        vmax_res = np.percentile(np.abs(residual), 98)
        plot_single(residual, f"{outdir}/res_iter={it}.png", "RdGy", vmin=-vmax_res, vmax=vmax_res, show_cbar=True)

        with torch.no_grad():
            g = v_est.grad.detach().cpu().numpy()
            # gmax_obs = np.percentile(np.abs(g), 99.5)
            # g_min = g.min()
            # g_max = g.max()
            plot_single(np.abs(g), f"{outdir}/grad_iter={it}.png", "magma", vmin=0, vmax=g.max(), show_cbar=True)


# # =====================================================
# # δv-based inversion (recommended)
# # v(x,z) = v_bg(x,z) + δv(x,z)
# # =====================================================

# # Initialize δv
# dv = torch.zeros_like(v_bg_t, requires_grad=True)

# optimizer = torch.optim.Adam([dv], lr=0.1)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

# losses = []


# def find_mute_index(y, tol=1e-6):
#     """
#     y: 2D RTM image (nx, ny)
#     tol: threshold on variance to detect first "non-mute" row
#     """
#     nx, ny = y.shape
#     for i in range(nx):
#         if np.var(y[i, :]) > tol:    # row starts having structure
#             return i
#     return nx  # all muted (unlikely)

# # usage
# mute_idx = find_mute_index(y_obs)   # or y_obs, y_pred
# print("mute index", mute_idx)


# for it in range(1, n_iter+1):
#     optimizer.zero_grad()

#     # Construct v = v_bg + dv
#     v_est = v_bg_t + dv

#     # Clamp to physical range without breaking graph
#     with torch.no_grad():
#         v_est.clamp_(v_min, v_max)

#     # Model input
#     x_in = torch.stack([v_est, v_bg_t], dim=0).unsqueeze(0)
#     y_pred = model(x_in).squeeze(0).squeeze(0)

#     # Data misfit
#     data_res  = y_pred - y_obs_t
#     # data_loss = torch.mean(data_res**2)
#     mask = torch.zeros_like(y_obs_t)
#     mask[mute_idx:, :] = 1.0     # only invert below mute depth
#     # print("y_pred.shape", y_pred.shape)
#     data_loss = torch.mean(( data_res * mask )**2)


#     # Regularization on δv only
#     tv  = tv_loss(ensure_4d(dv))
#     lap = laplacian_loss_aniso(dv.unsqueeze(0).unsqueeze(0))
#     l1  = torch.mean(torch.abs(dv))

#     # Total loss
#     loss = data_loss #+ lambda_tv*tv + lambda_l2*lap + lambda_l1*l1
#     loss.backward()

#     # Keep steps stable
#     # torch.nn.utils.clip_grad_norm_([dv], max_norm=1.0)

#     optimizer.step()
#     scheduler.step()

#     # Smooth δv update (keeps reflector structure, removes speckle)
#     with torch.no_grad():
#         dv[:] = gaussian_smooth_torch(dv.unsqueeze(0).unsqueeze(0), sigma=1.5).squeeze(0).squeeze(0)

#     # Logging
#     losses.append(loss.item())

#     if it % 2 == 0 or it == 1:
#         print(f"[{it:04d}/{n_iter}] loss={loss.item():.3e} data={data_loss.item():.3e} tv={tv.item():.3e} lap={lap.item():.3e} l1={l1.item():.3e}")

#         # Save velocity panel
#         plot_single(
#             v_est.detach().cpu().numpy(), 
#             f"{outdir}/vel_iter={it}.png", 
#             "turbo"
#         )

#         # Save forward RTM
#         y_np = y_pred.detach().cpu().numpy()
#         vmax_obs = np.percentile(np.abs(y_np), 98)
#         plot_single(
#             y_np, 
#             f"{outdir}/rtm_iter={it}.png", 
#             "gray", 
#             vmin=-vmax_obs, vmax=vmax_obs
#         )

#         # Save gradient
#         g = dv.grad.detach().cpu().numpy()
#         plot_single(
#             g, 
#             f"{outdir}/grad_iter={it}.png", 
#             "seismic", 
#             vmin=g.min(), vmax=g.max(),
#             show_cbar=True
#         )




# =====================================================
# Save results
# =====================================================
v_est_np = v_est.detach().cpu().numpy()
y_pred_np = y_pred.detach().cpu().numpy()

np.save(os.path.join(outdir, "v_est.npy"), v_est_np)
np.save(os.path.join(outdir, "v_bg.npy"), v_bg)
np.save(os.path.join(outdir, "v_true.npy"), v_true)
np.save(os.path.join(outdir, "y_obs.npy"), y_obs)
np.save(os.path.join(outdir, "y_pred.npy"), y_pred_np)
np.save(os.path.join(outdir, "losses.npy"), np.array(losses))

# =====================================================
# Plot diagnostics
# =====================================================

plt.figure(figsize=(6,4))
plt.semilogy(losses, marker="o", markevery=10, markersize=8)
plt.xlabel("Iteration")
plt.ylabel(r"$\mathcal{L}$")
plt.title("Least-squares inversion objective")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=200)
plt.close()



v_max= 4.763645
y_max= 10757.324219 / 500

v_bg = v_bg * v_max
v_est_np = v_est_np * v_max
v_true = v_true * v_max

# velocity
fig, axs = plt.subplots(1, 3, figsize=(25, 5))

# Compute global color limits
vmin = min(v_bg.min(), v_est_np.min(), v_true.min())
vmax = max(v_bg.max(), v_est_np.max(), v_true.max())

# Plot all panels with shared range
im0 = axs[0].imshow(v_bg, cmap="turbo", aspect="auto", vmin=vmin, vmax=vmax) #@TODO v_bg
axs[0].set_title("Background Velocity")

im1 = axs[1].imshow(v_est_np, cmap="turbo", aspect="auto", vmin=vmin, vmax=vmax)
axs[1].set_title("Recovered Velocity")

im2 = axs[2].imshow(v_true, cmap="turbo", aspect="auto", vmin=vmin, vmax=vmax)
axs[2].set_title("True Velocity")

for ax in axs:
    ax.axis("off")

# Reserve space for colorbar *before* adding it
fig.subplots_adjust(right=0.92, wspace=0.05)  # shrink main area a bit
# Add colorbar aligned to the right of the figure
cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(im2, cax=cbar_ax)
cbar.set_label("Velocity (km/s)")
plt.savefig(os.path.join(outdir, "velocity_compare.png"), dpi=200, bbox_inches="tight")
plt.close()



# RTM comparison
fig, axs = plt.subplots(1, 3, figsize=(18,5))
# Compute residual
residual = y_obs - y_pred_np
# Common color scale for obs/pred
vmax_obs = np.percentile(np.abs(y_obs), 98)
# Observed
axs[0].imshow(y_obs, cmap="gray", aspect="auto",
              vmin=-vmax_obs, vmax=vmax_obs)
axs[0].set_title("Groundtruth RTM")
# Predicted
axs[1].imshow(y_pred_np, cmap="gray", aspect="auto",
              vmin=-vmax_obs, vmax=vmax_obs)
axs[1].set_title("Predicted RTM (forward)")
# Residual
vmax_res = np.percentile(np.abs(residual), 98)
axs[2].imshow(residual, cmap="seismic", aspect="auto",
              vmin=0, vmax=vmax_res)
axs[2].set_title("Residual")

plt.tight_layout()
plt.savefig(os.path.join(outdir, "rtm_compare.png"), dpi=200)
plt.close()


# Velocity vertical trace
nx, ny = v_est_np.shape
print("nx", nx, ny) #256,512
mid_col = ny // 2
depth = np.arange(nx) * 12.5
plt.figure(figsize=(6,8))
plt.plot(v_filter[:, mid_col], depth, label="Background", color="b", linewidth=2, alpha=0.7)
plt.plot(v_est_np[:, mid_col], depth, label="Recovered", color="r", linewidth=2, alpha=0.7)
plt.plot(v_true[:, mid_col], depth, label="True", color="black", linewidth=2, linestyle="--")
plt.gca().invert_yaxis()
plt.xlabel("Velocity (km/s)")
plt.ylabel("Depth (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "velocity_trace.png"), dpi=200)
plt.close()

# RTM trace
mid_col = y_obs_t.shape[1] // 2
depth = np.arange(y_obs_t.shape[0])
# run forward model with current velocity estimate
x_in_trace = torch.stack([v_est, v_bg_t], dim=0).unsqueeze(0)
y_pred_trace = model(x_in_trace).squeeze(0).squeeze(0).detach().cpu().numpy()
y_obs_np = y_obs_t.cpu().numpy()

plt.figure(figsize=(6, 8))
plt.plot(y_obs_np[:, mid_col], depth, label="GroundTruth RTM", color="k", linewidth=2, linestyle="--")
plt.plot(y_pred_trace[:, mid_col], depth, label=f"Predicted (iter={it})", color="r", linewidth=2, alpha=0.7)
plt.gca().invert_yaxis()
plt.xlabel("Amplitude")
plt.ylabel("Depth (m)")
plt.legend()
plt.title(f"RTM Vertical Trace at iter={it}")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(outdir, f"trace_iter_{it:04d}.png"), dpi=200)
plt.close()
print(f"✅ Inversion complete. Results saved to {outdir}/")





# import imageio.v2 as imageio   # removes DeprecationWarning
# import glob, os
# from PIL import Image

# def make_movie(img_pattern, output_path, fps=5):
#     imgs = sorted(glob.glob(img_pattern))
#     print(f"Found {len(imgs)} frames for {output_path}")
#     if not imgs:
#         raise FileNotFoundError(f"No matching images for {img_pattern}")

#     # Load all frames, resize to first image size
#     frames = []
#     base_size = Image.open(imgs[0]).size
#     for img_path in imgs:
#         img = Image.open(img_path).convert("RGB")
#         if img.size != base_size:
#             img = img.resize(base_size, Image.LANCZOS)
#         frames.append(np.array(img))

#     imageio.mimsave(output_path, frames, fps=fps)
#     print(f"🎬 Saved movie to {output_path}")


# make_movie(os.path.join(outdir, "grad_iter=*.png"), os.path.join(outdir, "gradients_movie.gif"))
# make_movie(os.path.join(outdir, "obs_iter=*.png"), os.path.join(outdir, "rtms_movie.gif"))
# make_movie(os.path.join(outdir, "vel_iter=*.png"), os.path.join(outdir, "vel_movie.gif"))
