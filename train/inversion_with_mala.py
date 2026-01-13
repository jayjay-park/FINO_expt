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
from groundwater.devito_op import GroundwaterModel, GroundwaterLayer, GroundwaterEquation
from groundwater.utils import GaussianRandomField, plot_fields
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
def plot_single(true1, path, cmap="viridis", vmin=None, vmax=None):
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


def plot_inversion_result(x0, x, true_y, y, x_pred, loss_type, index, x_idx, y_idx, iter, folder):
    # pull everything off‐GPU, to numpy:
    fields = [
        x0.detach().squeeze().cpu().numpy(),       # initial guess
        true_y.squeeze().cpu().numpy(),            # ground truth output
        x.squeeze().cpu().numpy(),                 # ground truth input
        y.squeeze().cpu().numpy(),                 # forward prediction
        x_pred,    # inversion result
        np.abs(x.squeeze().cpu().numpy() - x_pred)
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
        data = fields[i].squeeze()
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
                cmap='viridis' if i<5 else 'magma',
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
        plt.savefig(f"{folder}/inversion_result_{loss_type}_{index}_{iter}.png")
    elif loss_type == "JAC" and top_subsampling == True:
        plt.savefig(f"{folder}_top/inversion_result_{loss_type}_{index}_{iter}.png")
    elif loss_type != "JAC" and top_subsampling == False:
        plt.savefig(f"{folder}/inversion_result_{loss_type}_{index}_{iter}.png")
    else:
        plt.savefig(f"{folder}_top/inversion_result_{loss_type}_{index}_{iter}.png")
    plt.close(fig)


def psd_1d(field: torch.Tensor,
           dim: int = -1,           # which spatial axis to transform
           norm_fft: str = "ortho"):
    """
    1-D power-spectral density of a tensor that may have batch / channel axes.

    field : (..., N) real-valued tensor
    dim   : axis along which to compute the FFT (default = last)
    """
    # promote to float32/64 if needed
    field = field.to(torch.get_default_dtype())

    # real-valued FFT is slightly faster and avoids negative frequencies
    F = torch.fft.rfft(field, dim=dim, norm=norm_fft)        # (..., N//2+1)
    power = (F.real**2 + F.imag**2)

    # average over every axis except the frequency axis
    freq_axis = dim if dim >= 0 else field.dim() + dim
    reduce_axes = tuple(i for i in range(power.dim()) if i != freq_axis)
    psd = power.mean(dim=reduce_axes)

    # wavenumber vector
    n = field.shape[dim]
    k = torch.fft.rfftfreq(n) * n            # 0, 1, …, N//2  (integer k)
    return k.to(field.device), psd

def psd_flattened(field: torch.Tensor, norm_fft="ortho"):
    """
    Flatten everything to 1-D and compute a PSD (rFFT).
    field: Tensor of shape (..., H, W)
    Returns k (int wavenumbers) and 1-D PSD averaged over leading dims.
    """
    B = field.reshape(-1, field.numel() // field.shape[0])  # (batch_like, N=H*W)
    F = torch.fft.rfft(B, dim=-1, norm=norm_fft)             # (batch_like, N/2+1)
    power = (F.real**2 + F.imag**2)
    psd = power.mean(dim=0)                                  # average across batch_like
    N = B.shape[-1]
    k = torch.fft.rfftfreq(N) * N                            # 0 … N/2
    return k.to(field.device), psd


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


from concurrent.futures import ThreadPoolExecutor, as_completed

def compute_gradient_single(simulator, x_np, probe_2d, p_fwd):
    """
    CPU worker: runs one Devito gradient (J^T v) and returns a flat array.
    """
    g2d = simulator.compute_gradient(x_np, probe_2d, p_fwd)  # [H,W]
    return g2d.reshape(-1)  # → (p,)


def fisher_approx_vjp_batched(
    model_or_simulator,
    x0,
    i,
    j,
    sigma,
    rank=200,
    chunk_size=32,
    loss_type="FNO",
    forcing_term=None
):
    device = x0.device
    p = x0.numel()       # total #parameters = H*W
    m = i.numel()        # #observations

    # 1) sample & orthonormalize all probes: V_full ∈ R^{m×r}
    if noise_std == 0:
        V_full = torch.randn(m, rank, device=device)
    else:
        V_full = (1.0 / sigma) * torch.randn(m, rank, device=device)
    V_full, _ = torch.linalg.qr(V_full)
    print("V", V_full.shape)

    # prepare result buffer
    Q = torch.empty(p, rank, device=device)

    # Devito: do one forward solve
    if loss_type == "Devito":
        x_np = x0.detach().cpu().numpy().squeeze()  # [H,W]
        f_np = (forcing_term.cpu().numpy()
                if torch.is_tensor(forcing_term)
                else forcing_term)
        p_fwd = model_or_simulator.eval_fwd_op(f_np, x_np, return_array=False)

    # FNO: define obs_model(flat_x) → [m]
    def obs_model(flat_x):
        x_in = flat_x.view_as(x0)
        out  = model_or_simulator(x_in)[0,0,:,:].flatten()  # [H*W]
        if noise_std == 0:
            return out[i]
        else:
            return out[i] / sigma                              # pick [m] entries

    # 2) chunk through all r probes
    for start in range(0, rank, chunk_size):
        end = min(start + chunk_size, rank)
        chunk_len = end - start
        V_chunk = V_full[:, start:end]    # [m, chunk_len]

        if loss_type != "Devito":
            # --- FNO path: batched VJP over axis=1 of V_chunk ---
            x_flat, pullback = torch.func.vjp(obs_model,
                                   x0.detach().clone().requires_grad_(True).flatten())

            with torch.no_grad():
                # map each column V_chunk[:,k] → pullback(v)[0], stacked along dim1
                # in_dims=1 says: input has shape [m,chunk]. map over chunk axis
                # out_dims=1: output shape will be [p,chunk]
                Q_chunk = torch.func.vmap(lambda v: pullback(v)[0],
                               in_dims=1, out_dims=1)(V_chunk)  # [p,chunk_len]
                print(Q_chunk.shape)
                Q[:, start:end] = Q_chunk

            del x_flat, pullback, Q_chunk
            torch.cuda.empty_cache()

        else:
            # --- Devito: sequential compute_gradient per probe ---
            H = x_np.shape[0]
            grads = []

            # for k in range(end - start):
                # build 2D probe
                # print("k", k)
            # probe_2d = np.zeros_like(x_np, dtype=np.float32)
            probe_2d = np.zeros((chunk_size, p))
            # flat_vec = V_chunk[:, k].detach().cpu().numpy()
            flat_vec = V_chunk.detach().cpu().numpy().reshape(chunk_size, -1)    
            probe_flat = np.zeros((chunk_size, p))   

            if full_obs == True:
                probe_2d = flat_vec
            else:
                idx = torch.tensor(L.detach().cpu(), dtype=torch.long)                
                rows = np.arange(chunk_size).reshape(-1, 1)  
                probe_2d[rows, idx] = flat_vec
            probe_2d = probe_2d.reshape(chunk_size, 128, 128)
            grads_np = np.empty((chunk_size, p), dtype=np.float32)
            with ThreadPoolExecutor() as exe:
                futures = {
                    exe.submit(compute_gradient_single, model_or_simulator, x_np, probe_2d[item_idx], p_fwd): item_idx
                    for item_idx in range(chunk_size)
                }
                for fut in as_completed(futures):
                    interm = futures[fut]
                    grads_np[interm] = fut.result()

            # compute Jᵀv via Devito
            # g2d = model_or_simulator.compute_gradient(x_np, probe_2d, p_fwd)
            grads.append(torch.from_numpy(grads_np).to(device).reshape(p, chunk_size))

            # stack into [p, chunk_len]
            Q[:, start:end] = torch.stack(grads)

            del grads
            torch.cuda.empty_cache()

    return Q

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
    posterior_set, output_set, curves = [], [], []
    # optimizer = torch.optim.Adam([x0], lr=learning_rate)

    losses, inversion_MSEs, regs, ssims, infty_norm = [], [], [], [], []
    loss_data_iter = []

    start_time = time.time()
    true_data = true_data.to(device)

    if loss_type == "Devito":
        if full_obs == True:
            extracted_target = true_data[:, :, i, j]
        else:
            extracted_target = true_data[i, j]
    else:
        extracted_target = true_data[:, :, i, j]
    plot_single(true_data.detach().cpu().squeeze(), "sanitycheck_y.png", "jet")



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

        if type_opt == "NGD":
            with torch.no_grad():
                if loss_type == "Devito":
                    Q = fisher_approx_vjp_batched(groundwater_model, x0, i, j, noise_std,rank=400,chunk_size=50,loss_type=loss_type,forcing_term=forcing_term)
                else:
                    Q = fisher_approx_vjp_batched(model, x0, i, j, noise_std, rank=400, chunk_size=100)#250

            # Precondition gradient
            '''g = x0.grad.detach().flatten()
            B = Q.T @ Q  # [r × r]
            natural_grad = Q @ torch.linalg.solve(B, Q.T @ g)
            g = natural_grad.reshape_as(x0)'''

            # ---- choose damping (can be constant or adaptive) ----
            lam = 1e-1                 # e.g. 1e-3; try 1e-4 … 1e-1 or make it adaptive
            g_flat   = x0.grad.detach().flatten()
            Q_t_g    = Q.T @ g_flat           # shape [r]
            B        = Q.T @ Q                # shape [r, r]
            # (B + lam I)^{-1} (Q^T g)
            w = torch.linalg.solve(B + lam * torch.eye(B.shape[0], device=B.device), Q_t_g)
            # component inside the rank-r Fisher subspace
            ng_sub   = Q @ w                  # shape [d]
            # component orthogonal to that subspace, scaled by 1/lam
            g_perp   = g_flat - Q @ (Q_t_g)   # (I − Q Qᵀ) g
            ng_perp  = g_perp / lam
            # full damped natural gradient
            g = (ng_sub + ng_perp).reshape_as(x0)



        if iteration % decay_interval == 0 and iteration > 0: #@TODO
            tuned_lr = two_sided_armijo_line_search(x0, loss_fn, g, learning_rate)
            learning_rate = tuned_lr  # update your base for the next block
            print("new lr", tuned_lr)

        with torch.no_grad():
            x0 -= learning_rate * g
            x0.data = torch.clamp(x0.data, min=0., max=1.0) #@TODO

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
            "loss":          loss.item(),
            "inversion_MSE": inversion_MSE.item(),
            "regularization":reg.item(),
            "SSIM":          ssim_value
        })

        # if batch_num < 2 and iteration % 50 == 0 and loss_type != "Devito":
        if batch_num < 2 and iteration % 50 == 0:
            gradient = x0.grad.detach().cpu().squeeze()  # shape: [H, W] or similar
            plt.imshow(gradient.numpy(), cmap='viridis')
            plt.colorbar(shrink=0.8)
            plt.tight_layout()
            plt.title('Gradient w.r.t. Input x0')

            if spectral == True:
                Δa   = (x0_new - x0_old).detach()    # the update field
                k, p = psd_flattened(Δa)
                curves.append((k.cpu().numpy(), p.cpu().numpy()))

            if loss_type == "JAC":
                folder = f'inversion_result_{loss_type}_{num_vec}_{initial_guess}_{type_opt}'
                plt.savefig(f'{folder}/iter={batch_num}_gradient_{iteration}.png')
                plot_single(x0.detach().cpu().squeeze(), f'{folder}/iter={batch_num}_inversion_{iteration}.png')
                plot_single(output.detach().cpu().squeeze(), f'{folder}/iter={batch_num}_inversion_{iteration}_output.png')
                plot_inversion_result(zero_X, x, y, output.detach().cpu().squeeze(), x0.clone().detach().cpu().numpy(), loss_type, batch_num, i, j, iteration, folder)
            else:
                folder = f'inversion_result_{loss_type}_{initial_guess}_{type_opt}'
                plt.savefig(f'{folder}/iter={batch_num}_gradient_{iteration}.png')
                plot_single(x0.detach().cpu().squeeze(), f'{folder}/iter={batch_num}_inversion_{iteration}.png')
                plot_single(output.detach().cpu().squeeze(), f'{folder}/iter={batch_num}_inversion_{iteration}_output.png')
                plot_inversion_result(zero_X, x, y, output.detach().cpu().squeeze(), x0.clone().detach().cpu().numpy(), loss_type, batch_num, i, j, iteration, folder)

        print(f"Iteration {iteration}, Loss: {loss_total.item():.4e}", inversion_MSE.item(), ssim_value)

        # store for plotting later
        losses.append(loss_total.item())
        inversion_MSEs.append(inversion_MSE.item())
        regs.append(reg.item())
        ssims.append(ssim_value)
        posterior_set.append(x0.clone().detach().cpu().numpy())
        output_set.append(output.clone().detach().cpu().numpy())

    return posterior_set, losses, inversion_MSEs, regs, ssims, output, loss_data_iter, i, j, curves, folder, output_set

# ----------------------
# Least Squares Posterior Estimation (with per-iteration timing)
# ----------------------
def least_squares_posterior_estimation(model, input_data, true_data, learning_rate,
                                       batch_num, num_iterations=500, prior=None, i=None, j=None):
    x0 = input_data.clone().detach().requires_grad_(True).to(device)
    posterior_set, output_set, curves = [], [], []
    mse_loss = torch.nn.MSELoss()

    losses, inversion_MSEs, regs, ssims, infty_norm = [], [], [], [], []
    loss_data_iter = []

    start_time = time.time()
    true_data = true_data.to(device)

    for t in range(num_iterations):
        output = model(x0)
        loss = mse_loss(output[:, :, i, j], true_data[:, :, i, j])
        loss.backward()
        with torch.no_grad():
            x0 -= learning_rate * x0.grad
        x0.requires_grad_(True)
        posterior_set.append(x0.detach().cpu())
        output_set.append(output.detach().cpu())

        elapsed = time.time() - start_time
        loss_data_iter.append({
            "sample": batch_num,
            "iteration": t,
            "elapsed_s": elapsed,
            "loss": loss.item()
        })

    return posterior_set, losses, inversion_MSEs, regs, ssims, output, loss_data_iter, i, j, curves, folder_name, output_set


# ------------------------------------------------------------
# MALA Posterior Estimation with Diagnostics
# ------------------------------------------------------------
def mala_posterior_estimation(model, input_data, true_data,
                              step_size, batch_num, num_samples=1000,
                              prior=None, i=None, j=None,
                              sigma_prior=0.5, alpha=0.0,
                              burn_in=200, thin_every=5,
                              diagnostics=True, output_dir="."):
    x0 = input_data.clone().detach().requires_grad_(True).to(device)
    true_data = true_data.to(device)
    posterior_set, output_set, loss_data_iter = [], [], []
    mse_loss = torch.nn.MSELoss()
    start_time = time.time()

    def log_posterior(a):
        if loss_type == "Devito":
            a = a.squeeze()
            a.retain_grad() 
            output = model(a)
            pred = output[i,j] 
        else: 
            output = model(a)
            pred = output[:, :, i, j]
        neg_log_likelihood = mse_loss(pred.squeeze(), true_data[:, :, i, j].squeeze())
        neg_log_prior = ((a.squeeze() - prior.squeeze())**2).sum() / (2 * sigma_prior**2)
        reg_term = alpha * 0.0  # optional smoothness penalty
        return -(neg_log_likelihood + neg_log_prior + reg_term)

    def transition_log_prob(x_from, x_to, grad_from):
        mu = x_from + 0.5 * step_size**2 * grad_from
        return -((x_to - mu)**2).sum() / (2 * step_size**2)

    accepted = 0
    for t in range(num_samples):
        lp = log_posterior(x0)
        x0.grad = None
        lp.backward()
        grad = x0.grad.clone()

        noise = torch.randn_like(x0)
        proposal = x0 + 0.5 * step_size**2 * grad + step_size * noise
        proposal = proposal.clamp(0.0, 1.0).detach().requires_grad_(True)

        lp_prop = log_posterior(proposal)
        proposal.grad = None
        lp_prop.backward()
        grad_prop = proposal.grad.clone()

        log_q_forward = transition_log_prob(x0, proposal, grad)
        log_q_reverse = transition_log_prob(proposal, x0, grad_prop)

        log_accept_ratio = lp_prop - lp + log_q_reverse - log_q_forward
        if torch.log(torch.rand(1)).item() < log_accept_ratio.item():
            x0 = proposal.clone().detach().requires_grad_(True)
            accepted += 1

        if t >= burn_in and (t - burn_in) % thin_every == 0:
            posterior_set.append(x0.detach().cpu())
            output_set.append(model(x0).detach().cpu())

        elapsed = time.time() - start_time
        loss_data_iter.append({
            "sample": batch_num,
            "iteration": t,
            "elapsed_s": elapsed,
            "log_posterior": lp.item(),
            "acceptance_rate": accepted / (t + 1)
        })

    print("accepted", accepted, "num_samples", num_samples)
    print(f"Final acceptance rate: {accepted / num_samples:.3f}")

    if diagnostics and len(posterior_set) > 0:
        trace_tensor = torch.stack(posterior_set)
        H, W = 128, 128

        def autocorrelation(x, max_lag=100):
            x = np.ravel(x)  # make sure it's 1D
            x = x - np.mean(x)
            result = np.correlate(x, x, mode='full')
            result = result[result.size // 2:]
            if result[0] == 0:
                return np.ones(max_lag) * np.nan
            result /= result[0]
            return result[:max_lag]

        def compute_ess(x):
            acf = autocorrelation(x, max_lag=100)
            if np.isnan(acf).any():
                return np.nan
            return len(x) / (1 + 2 * np.sum(acf[1:]))


        # trace_np = trace_tensor[:, 64, 64].numpy()
        trace_np = trace_tensor[:, 0, 0, 64, 64].numpy()  # access pixel (64,64)

        plt.figure()
        plt.plot(trace_np)
        plt.title("Trace plot at (64,64)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/trace_{batch_num}.png")
        plt.close()

        plt.figure()
        plt.hist(trace_np, bins=50, density=True)
        plt.title("Histogram at (64,64)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hist_{batch_num}.png")
        plt.close()

        acf = autocorrelation(trace_np, max_lag=100)
        plt.figure()
        plt.stem(acf)
        plt.title("Autocorrelation at (64,64)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/autocorr_{batch_num}.png")
        plt.close()

        ess_val = compute_ess(trace_np)
        print(f"ESS at (64,64): {ess_val:.1f}")

        ess_map = np.zeros((H, W))
        for y in range(H):
            for x in range(W):
                ess_map[y, x] = compute_ess(trace_tensor[:, 0, 0, y, x].numpy())

        plt.figure(figsize=(8, 6))
        im = plt.imshow(ess_map, cmap="viridis")
        plt.colorbar(im, label="ESS")
        plt.title("ESS Heatmap")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ess_map_{batch_num}.png")
        plt.close()

        np.save(f"{output_dir}/ess_map_{batch_num}.npy", ess_map)

    return posterior_set, output_set, loss_data_iter

# ------------------------------------------------------------
# Main Script for Inversion on Multiple Samples (batch_size=1)
# ------------------------------------------------------------

if __name__ == "__main__":
    # Set up device and random seed.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print(f"Using device: {device}")

    '''Experimental Factor'''
    num_vec = 400 #400
    loss_type = "JAC"  # or "JAC" "MSE" "Devito"
    GRF = 3
    alpha = 0. #0.05
    initial_guess = "prior_mean" # "smooth", "noisy", "naturalperturb"
    decay_interval = 20

    '''Type of Optimizer: GD, NGD, MALA'''
    type_opt = "MALA" #MALA

    '''Experimental Factor on Observation'''
    noise_std = 0.1 #0.08 #0.03
    sub_sampling = True
    top_subsampling = False
    full_obs = not sub_sampling
    ood_prior = False
    ood_tau = 5.00 #3.05 #3.05

    '''Ploting Factor'''
    spectral = False
    diagnostics = True
    
    print("type opt", type_opt)

    if initial_guess == "prior_mean":
        learning_rate = 0.05 #5000 #7200 #1200 #@TODO
        num_sample = 1 #1
        num_sample_prior = 30 # 30, 50 #5 #@TODO
        num_epoch = 1001 #1001
        offset=414 #130 #@TODO
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
    
    # ----------------------------------------------
    # Load configuration and dataset. and checkpoint
    # ----------------------------------------------
    if loss_type == "JAC" and num_vec == 1:
        config = "configs/eigenvectors/e=1.yaml"
        ckpt_path = "checkpoints/n=128_e=1_m=FNO_s=RFS_l=JAC_20250513_164312/n=128_e=1_m=FNO_s=RFS_l=JAC_epoch=249_val_rel_l2_loss=0.0006.ckpt"
    elif loss_type == "JAC" and num_vec == 50:
        config = load_config("configs/eigenvectors/e_50.yaml")
        ckpt_path = f"checkpoints/n=400_e=50_m=FNO_s=RFS_l=JAC_20250624_120949/n=400_e=50_m=FNO_s=RFS_l=JAC_epoch=199_val_rel_l2_loss=0.0206.ckpt"
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
        # ckpt_path = "checkpoints/DARCY_MSE/Darcy_training_epoch=249_val_rel_l2_loss=0.0009_MSE_May14.ckpt"
        ckpt_path = "checkpoints/Darcy_training_20250615_133632/Darcy_training_epoch=199_val_rel_l2_loss=0.0133.ckpt"
        # Numerical Simulator
        forcing_term = torch.zeros(128, 128)
        gw_torch_model = GroundwaterModel(forcing_term.shape[0])
    # Numerical Simulator
    forcing_term = torch.zeros(128, 128)
    gw_torch_model = GroundwaterModel(forcing_term.shape[0])

    # Surrogate model OR numerical simulator
    if loss_type != "Devito":
        model = NSModel.load_from_checkpoint(ckpt_path).eval().to(device)
        with open("rng_state_devito.pkl", "rb") as f:
            state = pickle.load(f)
            np.random.set_state(state["np_random_state"])
            random.setstate(state["random_state"])
    elif loss_type == "Devito":
        model = lambda x: gw_torch_model(x, forcing_term)
        groundwater_model = GroundwaterEquation(forcing_term.shape[0])

    # Load Data
    data_config = load_config("output/n=400_e=200_m=FNO_s=RFS_l=JAC_20250615_133916/config.yaml")
    dataset = get_dataset(data_config.experiment.dataset_type, data_config.data_settings)
    dataloader = dataset.get_dataloader(offset=414, limit=num_sample)
    prior_dataloader = dataset.get_dataloader(offset=414, limit=num_sample_prior)

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
    # Construct folder
    if loss_type != 'JAC':
        folder_name = f'inversion_result_{loss_type}_{initial_guess}_{type_opt}'
    else:
        folder_name = f'inversion_result_{loss_type}_{num_vec}_{initial_guess}_{type_opt}'
    os.makedirs(folder_name, exist_ok=True)


    if loss_type == "JAC" and top_subsampling == False :
        fname = f'inversion_history_{loss_type}_{num_vec}_{initial_guess}_{type_opt}.h5'
        fname_output = f'inversion_history_output_{loss_type}_{num_vec}_{initial_guess}_{type_opt}.h5'
    elif loss_type != "JAC" and top_subsampling == False :
        fname = f'inversion_history_{loss_type}_{initial_guess}_{type_opt}.h5'
        fname_output = f'inversion_history_output_{loss_type}_{initial_guess}_{type_opt}.h5'
    else:
        fname = f'inversion_history_{loss_type}_{initial_guess}_{type_opt}_top.h5'
        fname_output = f'inversion_history_output_{loss_type}_{initial_guess}_{type_opt}_top.h5'
    # If it already exists, delete it (and any stale lock)
    if os.path.exists(fname): os.remove(fname)
    if os.path.exists(fname_output): os.remove(fname_output)
    # Now create it
    h5_file = h5py.File(fname, 'w')
    h5_file_output = h5py.File(fname_output, 'w')
    num_samples = len(dataloader)
    dset = h5_file.create_dataset(
        'a', 
        shape=(num_samples, num_epoch, 128, 128),
        dtype='f4',
        compression='gzip',
        compression_opts=4,
        chunks=(1, num_epoch, 128, 128)  # chunk by sample
    )
    dset_output = h5_file_output.create_dataset(
        'u', 
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
            sum_x = x.sum(dim=0)
        prior_mean = sum_x / num_sample_prior
        prior_mean = prior_mean.unsqueeze(dim=1).detach()
        print("shape of prior mean: ", prior_mean.shape)

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
                i = torch.tensor(f["i"][:]).to(device).long()
                j = torch.tensor(f["j"][:]).to(device).long()

        # initial guess logic …
        if initial_guess == "smooth":
            zero_X = apply_gaussian_smoothing(x, kernel_size, sigma)
        elif initial_guess == "noisy":
            zero_X = x + torch.randn_like(x) * noise_std
        elif initial_guess == "prior_mean":
            zero_X = prior_mean

        # observation logic ...
        if ood_prior == True:
            # 1. define ood x
            if loss_type == "Devito":
                grf = GaussianRandomField(2, 128, alpha=2, tau=ood_tau)
                u_samples = grf.sample(1)
                # Sample random fields
                u_samples[u_samples>=0] = 0.9
                u_samples[u_samples<0] = 0.1
                x_prev = torch.tensor(u_samples[0])
                zero_X = (zero_X * num_sample_prior + x_prev.reshape(1, 1, 128, 128).cuda()) / (num_sample_prior + 1)
                # append data
                with h5py.File(f"grf_sample_data_{sample_counter}.h5", "a") as f:
                    f.create_dataset("ood_x", data=x_prev.detach().cpu().numpy())
            else:
                # load data
                with h5py.File(f"grf_sample_data_{sample_counter}.h5", "r") as f:
                    x_prev = torch.tensor(f["ood_x"][:]).to(device)
            # 2. create ood obs
            if noise_std == 0:
                y = gw_torch_model(x_prev.detach().squeeze(), forcing_term) 
            else:
                y = gw_torch_model(x_prev.detach().squeeze(), forcing_term) + torch.randn_like(x_prev.detach().squeeze()) * noise_std
            y = y.reshape(1, 1, 128, 128).float()
            plot_single(x_prev.detach().cpu().squeeze(), f"ood_x.png", "viridis")
            plot_single(y.detach().cpu().squeeze(), f"ood_y.png", "jet")
        else:
            plot_single(y.detach().cpu().squeeze(), "sanitycheck_y0.png")
            y = batch['y'].to(device) + torch.randn_like(x) * noise_std
            plot_single(y.detach().cpu().squeeze(), "sanitycheck_y2.png")

        plot_single(zero_X.detach().cpu().squeeze(), f"zero_X_sample_{sample_counter}.png", "viridis")

        if type_opt == "MALA":
            posterior_set, output_set, loss_data_iter = mala_posterior_estimation(
                model, zero_X, y,
                step_size=learning_rate, batch_num=sample_counter,
                num_samples=num_epoch, prior=x, i=i, j=j,
                sigma_prior=0.5, alpha=alpha,
                burn_in=200, thin_every=5,
                diagnostics=True, output_dir=folder_name
            )
            pred = output_set[-1]
            posterior_mean = torch.stack(posterior_set).mean(dim=0)
            print("posterior_mean", posterior_mean.shape, torch.stack(posterior_set).shape)

            # Save to file
            # np.save(f\"{output_dir}/posterior_mean_{batch_num}.npy\", posterior_mean.squeeze().cpu().numpy())
            plot_single(posterior_mean.squeeze().cpu().numpy(), f"{folder_name}/posterior_mean.png", "viridis")

        else:
            posterior_set, losses, inversion_MSEs, regs, ssims, pred, loss_data_iter, i_idx, j_idx, curves, folder, output_set = (
                least_squares_posterior_estimation(
                    model, zero_X, y,
                    learning_rate, batch_num=sample_counter,
                    num_iterations=num_epoch, prior=x, i=i, j=j
                )
            )

        

        # Plot the final inversion result.
        final_x0 = torch.tensor(posterior_set[-1]).cpu().numpy()  # just the col indicesr(posterior_set[-1]).detach()
        plot_inversion_result(zero_X, x, y, pred.detach(), final_x0, loss_type, sample_counter, i_idx, j_idx, num_epoch, folder)

        if spectral == True:
            if loss_type == "JAC":
                run_name = f"{loss_type}_{num_vec}_{initial_guess}_{type_opt}"   # e.g. "MSE" , "JVP50"
            else:
                run_name = f"{loss_type}_{initial_guess}_{type_opt}"   # e.g. "MSE" , "JVP50"
            k_all  = np.stack([c[0] for c in curves])   # shape (n_snapshots, n_bins)
            psd_all= np.stack([c[1] for c in curves])
            np.savez(f"psd_{run_name}.npz", k=k_all, psd=psd_all)

        # posterior_set is a list of length num_epoch, each an 128×128 numpy array. Write them into the HDF5 at [sample_counter, :, :, :]:
        arr = np.stack(posterior_set, axis=0).squeeze()   # shape (num_epoch,128,128)
        arr_output = np.stack(output_set, axis=0).squeeze()   # shape (num_epoch,128,128)
        dset[sample_counter, :, :, :] = arr
        dset_output[sample_counter, :, :, :] = arr_output

        # collect this sample’s iteration‐by‐iteration records
        loss_data_all.extend(loss_data_iter)
        sample_counter += 1

    # save to single CSV
    df = pd.DataFrame(loss_data_all)
    # Close the HDF5 file when you’re done:
    h5_file.close()
    h5_file_output.close()
    with h5py.File(fname, 'r') as f:
        print("On‑disk dataset shape is", f['a'].shape)

    # # Compute and print averaged SSIM and L2 misfit over all samples.
    # # @TODO I want to save it in some file.

    # # Save all loss and metric data to CSV.

    if loss_type == "JAC" and top_subsampling == False:
        csv_file = f"loss_statistics_multiple_samples_{loss_type}_{num_vec}_{initial_guess}_{type_opt}.csv"
    elif loss_type != "JAC" and top_subsampling == False:
        csv_file = f"loss_statistics_multiple_samples_{loss_type}_{initial_guess}_{type_opt}.csv"
    else:
        csv_file = f"loss_statistics_multiple_samples_{loss_type}_{initial_guess}_{type_opt}_top.csv"

    df.to_csv(csv_file, index=False)
    print(f"Loss data saved to {csv_file}")