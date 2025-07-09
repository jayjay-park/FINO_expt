import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from physicsnemo.models.fno import FNO
import h5py
from skimage.metrics import structural_similarity as ssim
from matplotlib import colors
from scipy.interpolate import interp1d
import time
import sys
import torch.nn.functional as F
from groundwater.devito_op import GroundwaterModel, GroundwaterLayer
import h5py
import pickle
import os
import random

from models.ns_inversion import NSModel  # Your model
from utils import get_dataset, load_config, get_model  # Your utils

NUM_PSEUDO_TIMESTEPS: int = 500000

# ----------------------
# Gaussian Smoothing Functions
# ----------------------
def gaussian_kernel(size: int, sigma: float):
    """Creates a 2D Gaussian kernel."""
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel = gauss[:, None] @ gauss[None, :]
    return kernel

def apply_gaussian_smoothing(batch_matrix: torch.Tensor, kernel_size: int, sigma: float):
    """Applies Gaussian smoothing to a batch of input matrices using a Gaussian kernel."""
    kernel = gaussian_kernel(kernel_size, sigma).to(batch_matrix.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: 1 x 1 x k x k
    kernel = kernel.expand(1, 1, kernel_size, kernel_size)

    original_min = batch_matrix.amin(dim=(-2, -1), keepdim=True)
    original_max = batch_matrix.amax(dim=(-2, -1), keepdim=True)
    original_range = original_max - original_min

    smoothed_batch = F.conv2d(batch_matrix, kernel, padding=kernel_size // 2, groups=1)
    smoothed_min = smoothed_batch.amin(dim=(-2, -1), keepdim=True)
    smoothed_max = smoothed_batch.amax(dim=(-2, -1), keepdim=True)
    smoothed_range = smoothed_max - smoothed_min

    rescaled_batch = (smoothed_batch - smoothed_min) / (smoothed_range + 1e-8) * original_range + original_min
    return rescaled_batch


laplacian_kernel = torch.tensor([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape [1,1,3,3]

def gradient_penalty(x):
    x = x.unsqueeze(1) if x.ndim == 3 else x  # ensure shape [B,1,H,W]
    weight = laplacian_kernel.to(x.device)
    lap = F.conv2d(x, weight, padding=1)
    return torch.mean(lap**2)


# ----------------------
# Plotting Functions
# ----------------------
def plot_single(true1, path, cmap="jet", vmin=None, vmax=None):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})
    # print("vmin", vmin, vmax)
    # if vmin != 0:
    #     norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if (vmin is not None and vmax is not None) else colors.CenteredNorm()
    # else:
    #     norm = colors.Normalize(vmin=vmin, vmax=vmax) if (vmin is not None and vmax is not None) else colors.CenteredNorm()
    
    fig, ax = plt.subplots()
    cax = ax.imshow(true1, cmap=cmap) #, norm=norm
    plt.colorbar(cax, ax=ax, fraction=0.045, pad=0.06)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_observed_only_with_scatter(data, x_idx, y_idx, ax, cmap='jet'):
    # extract the value at each observation
    vals = data[y_idx, x_idx]

    # choose same norm logic
    vmin, vmax = vals.min(), vals.max()
    if vmin < 0 < vmax:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    sc = ax.scatter(x_idx, y_idx, c=vals, cmap=cmap, norm=norm, s=50, marker='s')
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    return sc


def plot_inversion_result(x0, x, true_y, y, x_pred, loss_type, index, x_idx, y_idx, iter):
    # pull everything off‐GPU, to numpy:
    fields = [
        x0.detach().squeeze().cpu().numpy(),       # initial guess
        true_y.squeeze().cpu().numpy(),            # ground truth output
        x.squeeze().cpu().numpy(),                 # ground truth input
        y.squeeze().cpu().numpy(),                 # forward prediction
        x_pred.detach().squeeze().cpu().numpy(),    # inversion result
        np.abs(x.squeeze().cpu().numpy() - x_pred.detach().squeeze().cpu().numpy())
    ]
    titles = [
        r'Initial Guess ($a_0$)',
        r'Observation ($y$)',
        r'Ground Truth Input ($a^\ast$)',
        r'Forward Prediction ($\hat{u}$)',
        r'Inversion Result ($a$)',
        r'$|a - a^\ast|$'
    ]

    # your observed locations
    x_idx = x_idx.detach().cpu().numpy()
    y_idx = y_idx.detach().cpu().numpy()

    fig, axes = plt.subplots(3, 2, figsize=(10,15))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        data = fields[i]
        # choose norm
        if i == 4:
            vmin = np.percentile(fields[2], 0.01)
            vmax = np.percentile(fields[2], 99.99)            
        else:
            vmin = np.percentile(data, 0.01)
            vmax = np.percentile(data, 99.99)
        print("vmin", vmin, "vmax", vmax)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        if i in (1, 3):  # only observed points
            if sub_sampling == True:
                sc = ax.scatter(x_idx, y_idx, c=data[y_idx, x_idx], cmap='jet', norm=norm, s=10, marker='o')
            else:
                sc = ax.scatter(x_idx, y_idx, c=data[y_idx, x_idx], cmap='jet', norm=norm, s=5, marker='o')
            mappable = sc
            # set the axes limits to match the image‐grid
            ax.set_xlim(-0.5, data.shape[1]-0.5)
            ax.set_ylim(data.shape[0]-0.5, -0.5)      # flip y so origin matches imshow
            ax.set_aspect('equal')

        else:  # full‐field image
            im = ax.imshow(
                data,
                cmap='jet' if i<5 else 'magma',
                norm=norm,
                origin='lower',
                extent=(0, data.shape[1], 0, data.shape[0]),
                aspect='equal'
            )
            mappable = im

        ax.set_title(titles[i])
        # exactly one colorbar:
        fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if loss_type == "JAC" and top_subsampling == False:
        plt.savefig(f"inversion_result_{loss_type}_{num_vec}_{initial_guess}/inversion_result_{loss_type}_{index}_{iter}.png")
    elif loss_type == "JAC" and top_subsampling == True:
        plt.savefig(f"inversion_result_{loss_type}_{num_vec}_{initial_guess}_top/inversion_result_{loss_type}_{index}_{iter}.png")
    elif loss_type != "JAC" and top_subsampling == False:
        plt.savefig(f"inversion_result_{loss_type}_{initial_guess}/inversion_result_{loss_type}_{index}_{iter}.png")
    else:
        plt.savefig(f"inversion_result_{loss_type}_{initial_guess}_top/inversion_result_{loss_type}_{index}_{iter}.png")
    plt.close(fig)


def radial_psd(field: torch.Tensor,
               n_bins: int | None = None,
               norm_fft: str = "ortho",
               return_counts: bool = False):
    """
    Compute the azimuthally-averaged (radial) power spectral density of a 2-D
    field or batch of fields.

    Parameters
    ----------
    field : Tensor
        Shape (B, C, H, W) **or** (C, H, W) **or** (H, W); real-valued.
    n_bins : int, optional
        Number of radial bins (default = Nyquist = min(H, W)//2 + 1).
    norm_fft : {"backward", "ortho", "forward"}, optional
        Normalisation passed to torch.fft.fftn.
    return_counts : bool, optional
        If True also return the number of frequency cells per bin.

    Returns
    -------
    k      : Tensor, shape (n_bins,)
        Radial wavenumber (0, 1, …).
    psd    : Tensor, shape (n_bins,)
        Mean power at each radial wavenumber.
    counts : Tensor, shape (n_bins,), optional
        Number of cells that contributed to each bin.
    """
    # -------- ensure a 4-D tensor (B, C, H, W) ---------------------------
    if field.dim() == 2:             # (H, W)
        field = field.unsqueeze(0).unsqueeze(0)
    elif field.dim() == 3:           # (C, H, W)
        field = field.unsqueeze(0)
    B, C, H, W = field.shape
    device     = field.device

    # --- pick a safe number of bins -----------------------------------
    if n_bins is None:
        # exact maximum radius that can appear on the centred grid
        max_r = int(torch.ceil(
            torch.sqrt(torch.tensor((H // 2) ** 2) + ((W // 2) ** 2))
        ).item()) + 1                   # +1 to include Nyquist shell
        n_bins = max_r

    # --- FFT & power as before ---
    F     = torch.fft.fftn(field, dim=(-2, -1), norm=norm_fft)
    power = F.real**2 + F.imag**2
    Pmean = power.mean(dim=(0, 1))

    # --- radius map ---------------------------------------------------
    ys = torch.arange(H, device=device) - H // 2
    xs = torch.arange(W, device=device) - W // 2
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    R    = torch.sqrt(Y**2 + X**2).round().long()

    R.clamp_(max=n_bins - 1)            # <─ keep indices legal

    # --- binning -------------------------------------------------------
    psd    = torch.zeros(n_bins, device=device)
    counts = torch.zeros(n_bins, device=device)
    psd.scatter_add_(0, R.flatten(), Pmean.flatten())
    counts.scatter_add_(0, R.flatten(), torch.ones_like(Pmean).flatten())
    psd /= counts.clamp_min(1)

    k = torch.arange(n_bins, device=device)
    return (k, psd, counts) if return_counts else (k, psd)


def apply_neumann_bc(u_interior):
    """
    Same as above but for a PyTorch tensor.
    """
    N, M = 126, 126
    u = u_interior.new_zeros((N+2, M+2))
    u[1:-1, 1:-1] = u_interior.squeeze()[1:-1, 1:-1]

    # top row:  
    u[0, 1:-1]  = 2*u[1, 1:-1] - u[2, 1:-1]  
    # bottom row:  
    u[-1, 1:-1] = 2*u[-2, 1:-1] - u[-3, 1:-1]  
    # left col:  
    u[1:-1, 0]  = 2*u[1:-1, 1]   - u[1:-1, 2]  
    # right col:  
    u[1:-1, -1] = 2*u[1:-1, -2]  - u[1:-1, -3]  


    u[0, 0]      = u[1, 1]
    u[0, -1]     = u[1, -2]
    u[-1, 0]     = u[-2, 1]
    u[-1, -1]    = u[-2, -2]

    return u.reshape(1, 1, 128, 128)

def backtracking_step(x0, loss_fn, mixed_grad, base_lr,
                      c=1e-4, beta=0.95, max_ls_iters=5):
    loss0 = loss_fn(x0)
    g_dot = torch.dot(mixed_grad.flatten(), mixed_grad.flatten())
    step = base_lr

    for _ in range(max_ls_iters):
        x_cand   = x0 - step * mixed_grad
        loss_cand = loss_fn(x_cand)
        if loss_cand <= loss0 - c * step * g_dot:
            print(loss_cand,  loss0, c * step * g_dot)
            return step      # <— only the step size
        step *= beta

    return base_lr            # fallback

def two_sided_armijo_line_search(x, loss_fn, grad, step0,
                       c=5e-1, beta=0.5, gamma=1.02,
                       max_expand=5, max_shrink=5):
    """Return a step size satisfying Armijo:  f(x-αg) ≤ f(x) - c α ||g||²."""
    f0   = loss_fn(x)
    gdot = torch.dot(grad.flatten(), grad.flatten()).item()

    # ---------- expansion phase ----------
    step = step0
    for _ in range(max_expand):
        f_try = loss_fn(x - step * grad)
        if f_try <= f0 - c * step * gdot:
            step *= gamma        # keep expanding while Armijo still holds
        else:
            step /= gamma        # last good step
            break

    # ---------- back-tracking phase ----------
    for _ in range(max_shrink):
        f_try = loss_fn(x - step * grad)
        if f_try <= f0 - c * step * gdot:
            return step
        step *= beta             # shrink
    return step0                 # fallback


def total_variance(x):
    return torch.mean(torch.abs(x[...,:-1] - x[...,1:])) + torch.mean(torch.abs(x[...,:-1,:] - x[...,1:,:]))

# ----------------------
# Least Squares Posterior Estimation (with per-iteration timing)
# ----------------------
def least_squares_posterior_estimation(model, input_data, true_data, learning_rate,
                                       batch_num, num_iterations=500, prior=None, i=None, j=None):
    if loss_type != "Devito":
        model.eval()
    mse_loss = torch.nn.MSELoss()

    x0 = input_data.clone().detach().requires_grad_(True).to(device)
    posterior_set, curves = [], []
    # optimizer = torch.optim.Adam([x0], lr=learning_rate)

    losses, inversion_MSEs, regs, ssims, infty_norm = [], [], [], [], []
    loss_data_iter = []

    start_time = time.time()

    if ood_prior == True:
        extracted_target = true_data[i, j]
    else:
        extracted_target = true_data[:, :, i, j]
    plot_single(true_data.detach().cpu().squeeze(), "sanitycheck_y.png")



    for iteration in range(num_iterations):
        # optimizer.zero_grad()
        x0.grad = None
        x0_old = x0.detach().clone()
        if loss_type == "Devito":
            squeezed_x0 = x0.squeeze()
            squeezed_x0.retain_grad()
            output = model(squeezed_x0)
        else:
            output = model(x0)

        # extract and compute loss
        if loss_type == "Devito":
            extracted_output = output[i, j]
        else:
            extracted_output = output[:, :, i, j]
            print("extracted output", extracted_output.shape)

        def loss_fn(x):
            """
            x: torch.Tensor shaped like x0 (e.g. [1,1,H,W]) — assumed on the correct device.
            Returns: scalar float = MSE(observed) + α·gradient_penalty(x)
            """
            with torch.no_grad():
                # 1) forward
                # target = true_data[0, 0, i, j]
                if loss_type == "Devito":
                    # Devito model expects [H,W] → returns [H,W]
                    out = model(x.squeeze())
                    # extract observed entries
                    pred = out[i, j]
                else:
                    # FNO / PyTorch path: [1,1,H,W] → [1,1,H,W]
                    out = model(x)
                    pred = out[0, 0, i, j]

                # 2) data misfit
                data_misfit = mse_loss(pred, extracted_target)

                # 4) combine and return as Python float
                return float(data_misfit.item())

        loss = mse_loss(extracted_output.squeeze(), extracted_target.squeeze())
        reg = total_variance(x0)
        loss_total = loss + alpha * reg
        loss_total.backward()
        g = x0.grad.detach()
        # optimizer.step()
        if iteration % decay_interval == 0 and iteration > 0:
            # re-tune the step once in a cheap way
            # tuned_lr = backtracking_step(
            #     x0, loss_fn, g, learning_rate,
            #     c=1e-4, beta=0.95, max_ls_iters=100
            # )
            tuned_lr = two_sided_armijo_line_search(x0, loss_fn, g, learning_rate)
            learning_rate = tuned_lr  # update your base for the next block
            print("new lr", tuned_lr)
        # just a fixed step
        with torch.no_grad():
            x0 -= learning_rate * g
            x0.data = torch.clamp(x0.data, min=0., max=1.0)
            # print("data", x0.data.shape)
            # x0.data[:, :, -1, :] = 0.1
            x0.requires_grad_(True)
            x0_new = x0.detach().clone()

        # metrics (L2)
        diff = x0 - prior
        inversion_MSE = torch.norm(diff, p=2) / torch.norm(prior)
        input_numpy = x0.detach().cpu().squeeze().numpy()
        prior_numpy = prior.detach().cpu().squeeze().numpy()
        ssim_value = ssim(input_numpy.astype(np.float64),
                          prior_numpy.astype(np.float64),
                          data_range=float(input_numpy.max() - input_numpy.min()))

        # elapsed time
        now = time.time()
        elapsed = now - start_time

        # record
        loss_data_iter.append({
            "sample":        batch_num,
            "iteration":     iteration,
            "elapsed_s":     elapsed,
            "loss":          loss_total.item(),
            "inversion_MSE": inversion_MSE.item(),
            "regularization":reg.item(),
            "SSIM":          ssim_value
        })

        # if batch_num < 2 and iteration % 50 == 0 and loss_type != "Devito":
        if batch_num < 2 and iteration % 50 == 0:
            gradient = x0.grad.detach().cpu().squeeze()  # shape: [H, W] or similar
            plt.imshow(gradient.numpy(), cmap='jet')
            plt.colorbar(shrink=0.8)
            plt.tight_layout()
            plt.title('Gradient w.r.t. Input x0')

            if spectral == True:
                Δa   = (x0_new - x0_old).detach()    # the update field
                k, p = radial_psd(Δa)                # tensors on GPU
                curves.append((k.cpu().numpy(), p.cpu().numpy()))

            if loss_type == "JAC":
                plt.savefig(f'inversion_result_{loss_type}_{num_vec}_{initial_guess}/iter={batch_num}_gradient_{iteration}.png')
                plot_single(x0.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{num_vec}_{initial_guess}/iter={batch_num}_inversion_{iteration}.png')
                plot_single(output.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{num_vec}_{initial_guess}/iter={batch_num}_inversion_{iteration}_output.png')
                plot_inversion_result(zero_X, x, y, output.detach().cpu().squeeze(), x0.clone().detach().cpu(), loss_type, batch_num, i, j, iteration)
            elif top_subsampling == True:
                plt.savefig(f'inversion_result_{loss_type}_{initial_guess}_top/iter={batch_num}_gradient_{iteration}.png')
                plot_single(x0.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{initial_guess}_top/iter={batch_num}_inversion_{iteration}.png')
                plot_single(output.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{initial_guess}_top/iter={batch_num}_inversion_{iteration}_output.png')
                plot_inversion_result(zero_X, x, y, output.detach().cpu().squeeze(), x0.clone().detach().cpu(), loss_type, batch_num, i, j, iteration)
            else:
                plt.savefig(f'inversion_result_{loss_type}_{initial_guess}/iter={batch_num}_gradient_{iteration}.png')
                plot_single(x0.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{initial_guess}/iter={batch_num}_inversion_{iteration}.png')
                plot_single(output.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{initial_guess}/iter={batch_num}_inversion_{iteration}_output.png')
                plot_inversion_result(zero_X, x, y, output.detach().cpu().squeeze(), x0.clone().detach().cpu(), loss_type, batch_num, i, j, iteration)

        print(f"Iteration {iteration}, Loss: {loss_total.item():.4e}", inversion_MSE.item(), ssim_value)

        # store for plotting later
        losses.append(loss_total.item())
        inversion_MSEs.append(inversion_MSE.item())
        regs.append(reg.item())
        ssims.append(ssim_value)
        posterior_set.append(x0.clone().detach().cpu().numpy())

    return posterior_set, losses, inversion_MSEs, regs, ssims, output.detach().cpu().squeeze(), loss_data_iter, i, j, curves



# ----------------------
# Main Script for Inversion on Multiple Samples (batch_size=1)
# ----------------------
if __name__ == "__main__":
    # Set up device and random seed.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print(f"Using device: {device}")

    '''Experimental Factor'''
    num_vec = 50 #400
    loss_type = "MSE"  # or "JAC" "MSE" "Devito"
    GRF = 3
    alpha = 0. #0.05
    initial_guess = "prior_mean" # "smooth", "noisy", "naturalperturb"
    decay_interval = 20
    '''Experimental Factor on Observation'''
    noise_std = 0 #0.25 #0.5 #1.0 #0.3
    sub_sampling = False
    top_subsampling = False
    full_obs = True
    ood_prior = False
    '''Ploting Factor'''
    spectral = True
    

    if initial_guess == "prior_mean":
        learning_rate = 1200 #200 #0.001 # 0.0001 (grf, fullobs) #0.005 (noisy, fullobs) #0.00005  # Inversion learning rate.
        num_sample = 1 #1
        num_sample_prior = 5 #5
        num_epoch = 1001 #1001
        offset=130
    elif initial_guess == "smooth":
        learning_rate = 5 #1 # 0.0001 (grf, fullobs) #0.005 (noisy, fullobs) #0.00005  # Inversion learning rate.
        offset=128
        num_sample = 1
        num_sample_prior = 100
        num_epoch = 4000
        if GRF == 1:
            kernel_size = 45 #55 #(grf, fullobs)
            sigma = 10.0 #100.0 # (grf, fullobs)
        elif GRF == 2:
            kernel_size = 55
            sigma = 100.0
        elif GRF == 3:
            kernel_size = 19
            sigma = 500.0
    elif initial_guess == "noisy":
        learning_rate = 0.5 # 0.0001 (grf, fullobs) #0.005 (noisy, fullobs) #0.00005  # Inversion learning rate.
        offset=128
    elif initial_guess == "naturalperturb":
        learning_rate = 0.5
        num_epoch = 30000
        num_sample = 1
        num_sample_prior = 100
    elif initial_guess == "randomperturb":
        learning_rate = 0.5
        num_epoch = 30000
        num_sample = 1
        num_sample_prior = 100
    
    # Load configuration and dataset. and checkpoint
    if loss_type == "JAC" and num_vec == 1:
        config = "configs/eigenvectors/e=1.yaml"
        ckpt_path = "checkpoints/n=128_e=1_m=FNO_s=RFS_l=JAC_20250513_164312/n=128_e=1_m=FNO_s=RFS_l=JAC_epoch=249_val_rel_l2_loss=0.0006.ckpt"
    elif loss_type == "JAC" and num_vec == 10:
        config = "output/n=128_e=10_m=FNO_s=RFS_l=JAC_20250512_144619/config.yaml"
        ckpt_path = "checkpoints/n=128_e=10_m=FNO_s=RFS_l=JAC_20250512_144619/n=128_e=10_m=FNO_s=RFS_l=JAC_epoch=249_val_rel_l2_loss=0.0005.ckpt"
    elif loss_type == "JAC" and num_vec == 50:
        config = load_config("configs/eigenvectors/e_50.yaml")
        ckpt_path = f"checkpoints/n=400_e=50_m=FNO_s=RFS_l=JAC_20250624_120949/n=400_e=50_m=FNO_s=RFS_l=JAC_epoch=199_val_rel_l2_loss=0.0206.ckpt"
    elif loss_type == "JAC" and num_vec == 100:
        # config = load_config("configs/eigenvectors/e_100.yaml")
        # ckpt_path = f"checkpoints/DARCY_JAC_100/Darcy_training_epoch=249_val_rel_l2_loss=0.0022_JAC_May14.ckpt"
        config = load_config("configs/eigenvectors/e=100_data=270.yaml")
        ckpt_path = f"checkpoints/n=170_e=100_m=FNO_s=RFS_l=JAC_20250531_104031/n=170_e=100_m=FNO_s=RFS_l=JAC_epoch=187_val_rel_l2_loss=0.0001.ckpt"
    elif loss_type == "JAC" and num_vec == 128:
        config = load_config("configs/eigenvectors/e=100_data=270.yaml")
        ckpt_path = f"checkpoints/n=270_e=100_m=FNO_s=RFS_l=JAC_20250531_123550/n=270_e=100_m=FNO_s=RFS_l=JAC_epoch=146_val_rel_l2_loss=0.0004.ckpt"
    elif loss_type == "JAC" and num_vec == 200:
        config = load_config("configs/eigenvectors/e_200.yaml")
        ckpt_path = f"checkpoints/n=400_e=200_m=FNO_s=RFS_l=JAC_20250615_133916/n=400_e=200_m=FNO_s=RFS_l=JAC_epoch=190_val_rel_l2_loss=0.0170.ckpt"
    elif loss_type == "JAC" and num_vec == 400:
        config = load_config("configs/eigenvectors/e_400.yaml")
        ckpt_path = f"checkpoints/n=400_e=400_m=FNO_s=RFS_l=JAC_20250617_131205/n=400_e=400_m=FNO_s=RFS_l=JAC_epoch=299_val_rel_l2_loss=0.0156.ckpt"
    elif loss_type == "RAND":
        config = load_config("output/n=128_e=8_m=FNO_s=RAND_l=JAC_20250421_124311/config.yaml")
        ckpt_path = f"checkpoints/n=128_e=8_m=FNO_s=RAND_l=JAC_20250421_125959/last.ckpt"
    elif loss_type == "MSE":
        config = load_config("configs/darcy_MSE.yaml")
        ckpt_path = "checkpoints/DARCY_MSE/Darcy_training_epoch=249_val_rel_l2_loss=0.0009_MSE_May14.ckpt"

    # Numerical Simulator
    forcing_term = torch.zeros(128, 128)
    groundwater_model = GroundwaterModel(forcing_term.shape[0])

    # Surrogate model OR numerical simulator
    if loss_type != "Devito":
        model = NSModel.load_from_checkpoint(ckpt_path).eval().to(device)
        with open("rng_state_devito.pkl", "rb") as f:
            state = pickle.load(f)
            np.random.set_state(state["np_random_state"])
            random.setstate(state["random_state"])
    elif loss_type == "Devito":
        model = lambda x: groundwater_model(x, forcing_term)

    # Load Data
    if num_vec == 200 or num_vec == 400 or num_vec == 50:
        print("200!")
        data_config = load_config("output/n=400_e=200_m=FNO_s=RFS_l=JAC_20250615_133916/config.yaml")
        print(data_config.experiment.dataset_type, data_config.data_settings)
        dataset = get_dataset(data_config.experiment.dataset_type, data_config.data_settings)
        dataloader = dataset.get_dataloader(offset=414, limit=num_sample)
        prior_dataloader = dataset.get_dataloader(offset=414, limit=num_sample_prior)
    else:
        data_config = load_config("output/n=128_e=50_m=FNO_s=RFS_l=JAC_20250512_141821/config.yaml")
        dataset = get_dataset(data_config.experiment.dataset_type, data_config.data_settings)
        dataloader = dataset.get_dataloader(offset=offset, limit=num_sample)
        prior_dataloader = dataset.get_dataloader(offset=offset, limit=num_sample_prior)

    # Save RNG state
    with open("rng_state_devito.pkl", "wb") as f:
        pickle.dump({
            "np_random_state": np.random.get_state(),
            "random_state": random.getstate()
        }, f)

    print("Devito run: first np.random sample =", np.random.rand())
    print("Devito run: first random sample =", random.random())


    # Initialize a list to hold loss and metric data for each sample.
    loss_data_all = []
    metrics_all = []
    sample_counter = 0
    final_ssim_list = []
    final_l2_list = []


    if loss_type == "JAC" and top_subsampling == False :
        fname = f'inversion_history_{loss_type}_{num_vec}_{initial_guess}.h5'
    elif loss_type == "JAC" and top_subsampling == True:
        fname = f'inversion_history_{loss_type}_{num_vec}_{initial_guess}_top.h5'
    elif loss_type != "JAC" and top_subsampling == False :
        fname = f'inversion_history_{loss_type}_{initial_guess}.h5'
    else:
        fname = f'inversion_history_{loss_type}_{initial_guess}_top.h5'
    # If it already exists, delete it (and any stale lock)
    if os.path.exists(fname):
        os.remove(fname)

    # Now create it
    h5_file = h5py.File(fname, 'w')
    num_samples = len(dataloader)
    dset = h5_file.create_dataset(
        'a', 
        shape=(num_samples, num_epoch, 128, 128),
        dtype='f4',
        compression='gzip',
        compression_opts=4,
        chunks=(1, num_epoch, 128, 128)  # chunk by sample
    )

    # Compute prior mean
    if initial_guess == "prior_mean":
        for batch in prior_dataloader:
            x = batch['x'].to(device)
            print("x", x.shape)
            sum_x = x.sum(dim=0)
            print("x", x.shape)

        prior_mean = sum_x / num_sample_prior  # shape: [C, H, W]
        prior_mean = prior_mean.unsqueeze(dim=1).detach()
        print(prior_mean.shape)

    # Prepare CSV accumulators:
    loss_data_all = []
    sample_counter = 0

    for batch in dataloader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        V = batch['v'].to(device)
        L = batch['L'].view(-1).to(device)
        d = int(x.shape[-1])
        cols = torch.tensor([ (idx.item() // d, idx.item() % d) for idx in L ], device=device)
        i = cols[:, 0].long()
        j = cols[:, 1].long()

        if sub_sampling == True:
            mask = torch.zeros((128, 128), dtype=torch.bool)
            mask[i, j] = True
            coords = mask.nonzero(as_tuple=False)
            num_total = coords.shape[0]
            subsample_ratio = 1 #0.15
            num_subsample = int(subsample_ratio * num_total)
            indices = torch.randperm(num_total)[:num_subsample]
            selected_coords = coords[indices]
            subsampled_mask = torch.zeros_like(mask)
            subsampled_mask[selected_coords[:, 0], selected_coords[:, 1]] = True
            final_mask = mask & subsampled_mask

            i, j = final_mask.nonzero(as_tuple=True)
            count = final_mask.sum().item()
            print(f"Number of True values: {count}")
        elif top_subsampling == True:
            mask = torch.zeros((128, 128), dtype=torch.bool)
            mask[i, j] = True
            mask[1:, :] = False  # Clear everything except top row
            i, j = mask.nonzero(as_tuple=True)
            count = mask.sum().item()
            print(f"Number of True values: {count}")
        elif full_obs == True:
            # Full observation: all (i, j) in 128 × 128 grid
            print("in full obs")
            grid_size = 128
            i, j = torch.meshgrid(
                torch.arange(grid_size, device=device),
                torch.arange(grid_size, device=device),
                indexing='ij'
            )
            i = i.reshape(-1)
            j = j.reshape(-1)
            print("i", i)

        if loss_type == "Devito":
            # save data
            with h5py.File(f"grf_sample_data_{sample_counter}.h5", "w") as f:
                f.create_dataset("x", data=x.detach().cpu().numpy())
                f.create_dataset("y", data=y.detach().cpu().numpy())
                f.create_dataset("L", data=L.detach().cpu().numpy())
                f.create_dataset("i", data=i.detach().cpu().numpy())  # just the row indices
                f.create_dataset("j", data=j.detach().cpu().numpy())  # just the col indices
        else:
            # load data
            with h5py.File(f"grf_sample_data_{sample_counter}.h5", "r") as f:
                x = torch.tensor(f["x"][:]).to(device)
                y = torch.tensor(f["y"][:]).to(device)
                L = torch.tensor(f["L"][:]).to(device)
                i = torch.tensor(f["i"][:]).to(device).long()  # ← restore observation indices
                j = torch.tensor(f["j"][:]).to(device).long()



        # initial guess logic …
        if initial_guess == "smooth":
            zero_X = apply_gaussian_smoothing(x, kernel_size, sigma)
            # zero_X = torch.full_like(x, 0.4, device=device)
            # zero_X[..., 45:-45] = x.amax()
            # zero_X = zero_X.detach().cpu()
        elif initial_guess == "noisy":
            zero_X = x + torch.randn_like(x) * noise_std
        elif initial_guess == "prior_mean":
            zero_X = prior_mean
        elif initial_guess == "naturalperturb":
            perturbation = 0.8
            print("V", V[0].shape, V.shape)
            x_prev = x.clone().detach()
            for r in range(80):
                x_prev += perturbation * V[0, :, :, r] # choosing the largest vector
            zero_X = x_prev.detach()
            # length = 5
            # x_prev = x.clone().detach()
            # x_prev[:, :, 10:10 + length, 35:35 + length] = 0.7
            # x_prev[:, :, 70:70 + length, 90:90 + length] = 0.7
            # x_prev[:, :, 90:90 + length, 10:10 + length] = 0.7
            # x_prev[:, :, 30:30 + length, 100:100 + length] = 0.7
            # zero_X = x_prev.detach()
        elif initial_guess == "randomperturb":
            # '''Direction FINO never saw during training'''
            # # V = [v1, …, vk] flattened into shape [k, H*W]
            # w = torch.randn_like(v[0])
            # # make w orthogonal to all top k eigenvectors
            # for vi in v[:k]:
            #     w -= (w.flatten() @ vi.flatten()) * vi
            # w = w / w.norm()
            # x0 = (x_true + eps * w).detach().clone().requires_grad_(True)
            perturbation = 0.3
            x_prev = x.clone().detach()
            for r in range(80):
                x_prev += perturbation * torch.randn_like(x) # choosing the largest vector
            zero_X = x_prev.detach()

        if ood_prior == True:
            length = 45
            x_prev = x.clone().detach()
            x_prev[:, :, 10:10 + length, 10:10 + length] = -20
            x_prev[:, :, 70:70 + length, 50:50 + length] = -10
            # x_prev[:, :, 20:20 + length, 60:60 + length] = 10
            x_prev[:, :, 10:10 + length, 70:70 + length] = 25
            y_wonoise = groundwater_model(x_prev.detach().squeeze(), forcing_term) 
            y = y_wonoise + torch.randn_like(x_prev.detach().squeeze()) * noise_std
            plot_single(y_wonoise.detach().cpu().squeeze(), f"sanitycheck_y1.png")
            plot_single(x_prev.detach().cpu().squeeze(), f"ood_x.png", "jet")
            plot_single(y.detach().cpu().squeeze(), f"ood_y.png", "jet")
        else:
            plot_single(y.detach().cpu().squeeze(), "sanitycheck_y0.png")
            y = batch['y'].to(device) + torch.randn_like(x) * noise_std
            plot_single(y.detach().cpu().squeeze(), "sanitycheck_y2.png")

        plot_single(zero_X.detach().cpu().squeeze(), f"zero_X_sample_{sample_counter}.png", "jet")
        

        posterior_set, losses, inversion_MSEs, regs, ssims, pred, loss_data_iter, i_idx, j_idx, curves = (
            least_squares_posterior_estimation(
                model, zero_X, y,
                learning_rate, batch_num=sample_counter,
                num_iterations=num_epoch, prior=x, i=i, j=j
            )
        )

        # Plot the final inversion result.
        final_x0 = torch.tensor(posterior_set[-1]).detach()
        plot_inversion_result(zero_X, x, y, pred, final_x0, loss_type, sample_counter, i_idx, j_idx, num_epoch)

        if spectral == True:
            if loss_type == "JAC":
                run_name = f"{loss_type}_{num_vec}_{initial_guess}"   # e.g. "MSE" , "JVP50"
            else:
                run_name = f"{loss_type}_{initial_guess}"   # e.g. "MSE" , "JVP50"
            k_all  = np.stack([c[0] for c in curves])   # shape (n_snapshots, n_bins)
            psd_all= np.stack([c[1] for c in curves])
            np.savez(f"psd_{run_name}.npz", k=k_all, psd=psd_all)

        # posterior_set is a list of length num_epoch, each an 128×128 numpy array.
        # Write them into the HDF5 at [sample_counter, :, :, :]:
        arr = np.stack(posterior_set, axis=0).squeeze()   # shape (num_epoch,128,128)
        dset[sample_counter, :, :, :] = arr

        # collect this sample’s iteration‐by‐iteration records
        loss_data_all.extend(loss_data_iter)
        sample_counter += 1

    # save to single CSV
    df = pd.DataFrame(loss_data_all)
    # Close the HDF5 file when you’re done:
    h5_file.close()
    with h5py.File(fname, 'r') as f:
        print("On‑disk dataset shape is", f['a'].shape)

    # # Compute and print averaged SSIM and L2 misfit over all samples.
    # # @TODO I want to save it in some file.

    # # Save all loss and metric data to CSV.

    if loss_type == "JAC" and top_subsampling == False:
        csv_file = f"loss_statistics_multiple_samples_{loss_type}_{num_vec}_{initial_guess}.csv"
    elif loss_type == "JAC" and top_subsampling == True:
        csv_file = f"loss_statistics_multiple_samples_{loss_type}_{num_vec}_{initial_guess}_top.csv"
    elif loss_type != "JAC" and top_subsampling == False:
        csv_file = f"loss_statistics_multiple_samples_{loss_type}_{initial_guess}.csv"
    else:
        csv_file = f"loss_statistics_multiple_samples_{loss_type}_{initial_guess}_top.csv"

    df.to_csv(csv_file, index=False)
    print(f"Loss data saved to {csv_file}")