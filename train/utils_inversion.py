import torch
import torch.nn.functional as F
import math
import h5py
import matplotlib.pyplot as plt
import numpy as np
from groundwater.devito_op import GroundwaterModel, GroundwaterLayer, GroundwaterEquation
from groundwater.utils import GaussianRandomField, plot_fields

# ---------------------------------------------------------------------
# Fourier–KL reconstruction (exact Beskos–Lan–Marzouk form)
# ---------------------------------------------------------------------
# def reconstruct_fourier_field(theta, nx=128, ny=128,
#                               ncmp=10, s=1.1, alpha=1.0, sigma=1.0,
#                               device="cpu"):
#     theta = theta.detach().cpu().numpy().reshape(-1)
#     x = np.linspace(0, 1, nx)
#     y = np.linspace(0, 1, ny)
#     X, Y = np.meshgrid(x, y, indexing="ij")

#     u = np.zeros_like(X)
#     idx = 0
#     for i in range(1, ncmp + 1):
#         for j in range(1, ncmp + 1):
#             lam = sigma * (alpha + np.pi**2 * (i**2 + j**2)) ** (-s / 2)
#             phi = np.cos(np.pi * (i - 0.5) * X) * np.cos(np.pi * (j - 0.5) * Y)
#             u += lam * theta[idx] * phi
#             idx += 1

#     u_torch = torch.tensor(u, dtype=torch.float32, device=device)
#     k_torch = torch.exp(u_torch)
#     return u_torch, k_torch

import torch

def reconstruct_fourier_field(theta, nx=128, ny=128,
                              ncmp=10, s=1.1, alpha=1.0, sigma=1.0,
                              device="cpu"):
    """
    Fully differentiable PyTorch version of reconstruct_fourier_field.
    Supports autograd and can run on CUDA for optimization loops.
    """
    # Ensure tensor on correct device and shape
    theta = theta.to(device).reshape(-1)

    # Create coordinate grids (no gradient needed)
    x = torch.linspace(0, 1, nx, device=device)
    y = torch.linspace(0, 1, ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Initialize u
    u = torch.zeros_like(X)

    idx = 0
    for i in range(1, ncmp + 1):
        for j in range(1, ncmp + 1):
            lam = sigma * (alpha + torch.pi**2 * (i**2 + j**2)) ** (-s / 2)
            phi = torch.cos(torch.pi * (i - 0.5) * X) * torch.cos(torch.pi * (j - 0.5) * Y)
            u = u + lam * theta[idx] * phi
            idx += 1

    k = torch.exp(u)
    return u, k


def u_from_theta(theta, ncmp=4, nx=128, ny=128, s=1.1, alpha=1.0, sigma=1.0):
    """
    Reconstruct log-permeability field from KL/Fourier coefficients.
    θ ∈ ℝ^(ncmp^2)
    Returns u(x,y) on [0,1]^2 of shape [nx, ny]
    """
    theta = theta.view(ncmp, ncmp)
    x = torch.linspace(0, 1, nx, device=theta.device)
    y = torch.linspace(0, 1, ny, device=theta.device)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    u = torch.zeros_like(X)
    for i in range(1, ncmp + 1):
        for j in range(1, ncmp + 1):
            lam = sigma * (alpha + math.pi**2 * (i**2 + j**2)) ** (-s / 2)
            u += lam * theta[i - 1, j - 1] * torch.cos(math.pi * (i - 0.5) * X) * torch.cos(math.pi * (j - 0.5) * Y)
    return u

# ---------------------------------------------------------------------
# Observation geometry (36 points on circle r=0.5 + center)
# ---------------------------------------------------------------------
def circular_observation_points(n_obs=100, radius=0.5, center=(0.5, 0.5)):
    angles = np.linspace(0, 2 * np.pi, int(n_obs*0.7), endpoint=False)
    pts = np.stack([center[0] + radius * np.cos(angles),
                    center[1] + radius * np.sin(angles)], axis=1)
    angles = np.linspace(0, 2 * np.pi, int(n_obs*0.3), endpoint=False)
    pts_smaller = np.stack([center[0] + 0.25 * np.cos(angles),
                    center[1] + 0.25 * np.sin(angles)], axis=1)
    pts = np.vstack([pts, pts_smaller, center])  # include center point
    return pts.astype(np.float32)


# ---------------------------------------------------------------------
# Bilinear sampling of pressure at given points
# ---------------------------------------------------------------------
def sample_pressure_at_points(p_field, obs_pts):
    nx, ny = p_field.shape
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    vals = []
    for (xq, yq) in obs_pts:
        ix = np.clip(int(xq / dx), 0, nx - 2)
        iy = np.clip(int(yq / dy), 0, ny - 2)
        tx = (xq - xs[ix]) / dx
        ty = (yq - ys[iy]) / dy
        v = (
            (1 - tx) * (1 - ty) * p_field[ix, iy]
            + tx * (1 - ty) * p_field[ix + 1, iy]
            + (1 - tx) * ty * p_field[ix, iy + 1]
            + tx * ty * p_field[ix + 1, iy + 1]
        )
        vals.append(v)
    return np.array(vals, dtype=np.float32)


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



# ----------------------
# fft
# ----------------------

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



# ----------------------
# regularization term
# ----------------------

def make_wavenumbers(Nx: int, Ny: int, L1: float, L2: float, device=None, dtype=torch.float32):
    # frequencies in cycles per unit length
    fx = torch.fft.fftfreq(Nx, d=L1 / Nx, device=device)
    fy = torch.fft.rfftfreq(Ny, d=L2 / Ny, device=device)  # half spectrum on last axis

    # convert to angular wavenumbers (rad / unit length)
    kx = 2 * math.pi * fx
    ky = 2 * math.pi * fy

    # grid for rfft2 layout: shape [Nx, Ny//2 + 1]
    k1, k2 = torch.meshgrid(kx, ky, indexing='ij')  # k1 ≡ kx, k2 ≡ ky
    return k1.to(dtype), k2.to(dtype)

def matern_prior(x, mu, k1, k2, tau=3.0, alpha=2.0, lam=1e-3):
    d = (x - mu).squeeze(0).squeeze(0)
    Dh = torch.fft.rfft2(d)
    k2sym = (k1**2 + k2**2)                  # same shapes as your NS spectral arrays
    w = (k2sym + tau**2)**alpha
    return lam * (w * (Dh.conj()*Dh).real).mean()


laplacian_kernel = torch.tensor([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape [1,1,3,3]

def gradient_penalty(x):
    '''
    Laplacian term (for smoothness)
    '''
    x = x.unsqueeze(1) if x.ndim == 3 else x  # ensure shape [B,1,H,W]
    weight = laplacian_kernel.to(x.device)
    lap = F.conv2d(x, weight, padding=1)
    return torch.mean(lap**2)

def total_variance(x):
    return torch.mean(torch.abs(x[...,:-1] - x[...,1:])) + torch.mean(torch.abs(x[...,:-1,:] - x[...,1:,:]))

def sobolev_H1_norm(u: torch.Tensor,        # (..., H, W)
                    dx: float = 1.0,
                    dy: float = 1.0) -> torch.Tensor:
    """
    Finite-difference H¹(Ω) norm  ‖u‖² = ∫ (u² + |∇u|²) .
    Works on CPU/GPU, keeps leading batch dims.
    """
    # ---- central differences with Neumann padding ----
    # x-gradient
    grad_x = (F.pad(u, (0, 0, 1, 1), mode='replicate')[..., 2:, :] -
              F.pad(u, (0, 0, 1, 1), mode='replicate')[..., :-2, :]) / (2 * dy)
    # y-gradient
    grad_y = (F.pad(u, (1, 1, 0, 0), mode='replicate')[..., :, 2:] -
              F.pad(u, (1, 1, 0, 0), mode='replicate')[..., :, :-2]) / (2 * dx)

    return torch.mean(u.pow(2) + grad_x.pow(2) + grad_y.pow(2))



# ----------------------
# mask
# ----------------------

def darcy_mask1(x, a_max=0.9, a_min=0.1):
    # a_max = x.max()
    # a_min = x.min()
    # return 1 / (1 + torch.exp(-x))
    # return 0.5 * (a_max - a_min) * torch.tanh(x) + 0.5 * (a_max + a_min)
    return x

def darcy_mask2(x, a_max=0.9, a_min=0.1):
    # x = 1 / (1 + torch.exp(-x))  # sigmoid
    x_mean = x.mean()
    # a_max = x.max()
    # a_min = x.min()
    # x = 0.5 * (a_max - a_min) * torch.tanh(x) + 0.5 * (a_max + a_min)
    # x = apply_gaussian_smoothing(x.unsqueeze(0), kernel_size=3, sigma=3.0)
    x[x >= x_mean] = 0.9
    x[x < x_mean] = 0.1
    return x.squeeze()

# ----------------------
# Least Squares Posterior Estimation (PINO-style + adaptive binary)
# ----------------------
def least_squares_posterior_estimation(model, input_data, true_data, learning_rate,
                                       batch_num, num_iterations=500, prior=None, i=None, j=None, folder=None):

    device_ = input_data.device
    mse_loss = torch.nn.MSELoss()

    '''
    Previous version
    '''

    # ---- global knobs (fall back if absent) ----
    lt   = globals().get("opt_type", "JAC")         # "Devito" or "JAC"
    dt   = globals().get("data_type", "Darcy")       # "Darcy"/"NS"
    full = globals().get("full_obs", True)           # use full-field loss if True
    do_ngd = (globals().get("type_opt", "GD") == "NGD")
    decay_interval = globals().get("decay_interval", 50)
    noise_std = globals().get("noise_std", 1e-3)
    forcing_term = globals().get("forcing_term", None)
    initial_guess = globals().get("initial_guess", "init")

    # ---- masks / binarization helpers ----
    a_min, a_max = 0.1, 0.9          # physical permeability range
    raw_clamp_min, raw_clamp_max = -6.0, 6.0  # clamp only in RAW/logit space

    def darcy_mask_tanh(x):
        return 0.5 * (a_max - a_min) * torch.tanh(x) + 0.5 * (a_max + a_min)

    def _kmeans2_threshold(x_phys, iters=10):
        v = x_phys.detach().flatten()
        if v.numel() == 0:
            return torch.tensor(0.5*(a_min+a_max), device=x_phys.device, dtype=x_phys.dtype)
        c1 = torch.quantile(v, 0.2)
        c2 = torch.quantile(v, 0.8)
        for _ in range(iters):
            d1 = (v - c1).abs()
            d2 = (v - c2).abs()
            m1 = v[d1 <= d2]
            m2 = v[d2 <  d1]
            if len(m1) == 0 or len(m2) == 0:
                break
            c1_new = m1.mean()
            c2_new = m2.mean()
            if torch.isclose(c1, c1_new) and torch.isclose(c2, c2_new):
                break
            c1, c2 = c1_new, c2_new
        return 0.5 * (c1 + c2)

    def binarize_from_phys(x_phys, method="kmeans", mid=None):
        if method == "kmeans":
            mid_val = _kmeans2_threshold(x_phys) if mid is None else torch.tensor(mid, device=x_phys.device, dtype=x_phys.dtype)
        elif method == "median":
            mid_val = torch.median(x_phys) if mid is None else torch.tensor(mid, device=x_phys.device, dtype=x_phys.dtype)
        else:
            mid_val = torch.tensor(0.5*(a_min+a_max), device=x_phys.device, dtype=x_phys.dtype)
        a_min_t = torch.tensor(a_min, device=x_phys.device, dtype=x_phys.dtype)
        a_max_t = torch.tensor(a_max, device=x_phys.device, dtype=x_phys.dtype)
        return torch.where(x_phys >= mid_val, a_max_t, a_min_t)

    def majority_filter_3x3(x_bin):
        mid = 0.5 * (a_min + a_max)
        x01 = (x_bin > mid).float()  # [1,1,H,W]
        kernel = torch.ones((1,1,3,3), device=x_bin.device)
        cnt = F.conv2d(x01, kernel, padding=1)
        maj = (cnt >= 5).float()
        a_min_t = torch.tensor(a_min, device=x_bin.device, dtype=x_bin.dtype)
        a_max_t = torch.tensor(a_max, device=x_bin.device, dtype=x_bin.dtype)
        return torch.where(maj > 0, a_max_t, a_min_t)

    posterior_set, output_set, curves = [], [], []
    losses, inversion_MSEs, regs, ssims, infty_norm, gradient_set = [], [], [], [], [], []
    loss_data_iter = []

    # ---- inputs / dtypes ----
    x_in = input_data.clone().detach().to(device_)
    y_in = true_data.clone().detach().to(device_)

    if lt == "Devito" and dt == "NS":
        x_in = x_in.double()
        y_in = y_in.double()

    # targets for pixel vs full-field
    if lt == "Devito" and dt == "Darcy":
        target_full = y_in if full else None
        target_pix  = y_in[:, :, i, j] if (y_in.ndim == 4) else y_in[i, j]
    else:
        target_full = y_in
        target_pix  = y_in[:, :, i, j]

    # sanity snapshot
    try:
        plot_single(y_in[0].detach().cpu().squeeze(), "sanitycheck_y.png", "jet")
    except Exception:
        pass

    # FREE/logit variable
    x_free = x_in.clone().detach().requires_grad_(True)

    # closure for line search
    def loss_fn(x_raw):
        with torch.no_grad():
            x_phys_ls = darcy_mask_tanh(x_raw)
            if lt == "Devito" and dt == "Darcy":
                out = model(x_phys_ls.squeeze())
                pred, tgt = (out, target_full) if full else (out[i, j], target_pix)
            else:
                out = model(x_phys_ls)
                pred, tgt = (out, target_full) if full else (out[0, 0, i, j], target_pix)
            return float(mse_loss(pred, tgt).item())

    start_time = time.time()

    for iteration in range(num_iterations):
        if wrt_NS:
            # Load simulator iterate for this sample and iteration
            x0_np = dset_ns[batch_num, iteration, :, :]   # numpy (128,128)
            x_free = torch.tensor(x0_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            x_free.requires_grad_(True)

        if iteration > 0:
            x_free = x_free.detach().clone().requires_grad_(True)
        x_free.grad = None

        # physical mapping
        x_phys = darcy_mask_tanh(x_free)
        x_phys.retain_grad()

        # forward
        if lt == "Devito" and dt == "Darcy":
            output = model(x_phys.squeeze())     # [H,W]
        else:
            output = model(x_phys)               # [B,C,H,W]

        # loss (full vs pixel)
        if lt == "Devito" and dt == "Darcy":
            pred_obs, tgt = (output, target_full) if full else (output[i, j], target_pix)
        else:
            pred_obs, tgt = (output, target_full) if full else (output[:, :, i, j], target_pix)

        loss_total = mse_loss(pred_obs, tgt)
        loss_total.backward()

        # gradient wrt FREE
        g = x_free.grad.detach()
        g_phys = x_phys.grad.detach()

        # NGD (optional)
        if do_ngd:
            with torch.no_grad():
                if lt == "Devito" and dt == "Darcy":
                    Q = fisher_approx_vjp_batched(
                        groundwater_model, x_phys, i, j, noise_std,
                        rank=400, chunk_size=50, opt_type=lt, forcing_term=forcing_term
                    )
                else:
                    Q = fisher_approx_vjp_batched(
                        model, x_phys, i, j, noise_std,
                        rank=400, chunk_size=100
                    )
            lam = 1e-1
            g_flat = g.flatten()
            Qtg    = Q.T @ g_flat
            B      = Q.T @ Q
            w      = torch.linalg.solve(B + lam * torch.eye(B.shape[0], device=B.device), Qtg)
            ng_sub  = Q @ w
            g_perp  = g_flat - Q @ Qtg
            ng_perp = g_perp / lam
            g = (ng_sub + ng_perp).reshape_as(x_free)

        # line search
        if iteration % decay_interval == 0 and iteration > 0:
            learning_rate = two_sided_armijo_line_search(x_free, loss_fn, g, learning_rate)
            print("new lr", learning_rate)
            # gradient wrt FREE
            g = x_free.grad.detach()

        # raw-space update
        with torch.no_grad():
            x_free -= learning_rate * g
            x_free.clamp_(raw_clamp_min, raw_clamp_max)
            x_free.requires_grad_(True)

        # ---- evaluation / logging ----
        with torch.no_grad():
            x_phys_eval = darcy_mask_tanh(x_free.detach())
            out_eval = (model(x_phys_eval) if not (lt == "Devito" and dt == "Darcy")
                        else model(x_phys_eval.squeeze()))

            # adaptive binary from PHYSICAL + small denoise
            x_bin_eval = binarize_from_phys(x_phys_eval, method="kmeans")
            x_bin_eval = majority_filter_3x3(x_bin_eval)

            # param metrics
            diff = x_phys_eval - prior
            inversion_MSE = torch.norm(diff, p=2) / (torch.norm(prior, p=2) + 1e-12)
            inv_mse = mse_loss(x_phys_eval, prior)

            # H1 / SSIM if available
            try:
                sobolev_num   = sobolev_H1_norm(diff)
                sobolev_denom = sobolev_H1_norm(prior)
                rel_sobolev   = (sobolev_num / (sobolev_denom + 1e-12)).sqrt().item()
            except Exception:
                rel_sobolev = float('nan')
            try:
                ssim_value = ssim(
                    x_bin_eval.detach().cpu().squeeze().numpy().astype(np.float64),
                    prior.detach().cpu().squeeze().numpy().astype(np.float64),
                    data_range=float(x_bin_eval.max().item() - x_bin_eval.min().item() + 1e-12)
                )
            except Exception:
                ssim_value = float('nan')

        # record line
        elapsed = time.time() - start_time
        loss_data_iter.append({
            "sample":        batch_num,
            "iteration":     iteration,
            "elapsed_s":     elapsed,
            "loss":          float(loss_total.item()),
            "inversion_MSE": float(inversion_MSE.item()),
            "regularization": 0.0,
            "SSIM":          ssim_value,
            "rel_H1":        rel_sobolev
        })

        # periodic plots (keep your originals + new binary)
        if batch_num < 6 and iteration % 100 == 0:
            try:
                # gradient image from g
                grad_img = g.detach().cpu().squeeze()
                plt.imshow(grad_img.numpy(), cmap='viridis'); plt.colorbar(shrink=0.8); plt.tight_layout()
                plt.savefig(f'{folder}/iter={batch_num}_gradient_{iteration}.png'); plt.close()
            except Exception:
                pass

            # your existing plots
            try:
                plot_single(x_phys_eval.detach().cpu().squeeze(), f'{folder}/iter={batch_num}_inversion_{iteration}.png')
                out_eval_tensor = out_eval if isinstance(out_eval, torch.Tensor) else torch.tensor(out_eval)
                plot_single(out_eval_tensor.detach().cpu().squeeze(), f'{folder}/iter={batch_num}_inversion_{iteration}_output.png')
            except Exception:
                pass

            # NEW binary plot
            plot_single(x_bin_eval.detach().cpu().squeeze(), f'{folder}/iter={batch_num}_binary_{iteration}.png', "viridis")
            plot_single(x_free.detach().cpu().squeeze(), f'{folder}/iter={batch_num}_free_{iteration}.png', "viridis")


            # keep your composite plot call if available (guarded)
            try:
                out_eval_tensor = out_eval if isinstance(out_eval, torch.Tensor) else torch.tensor(out_eval)
                plot_inversion_result(
                    zero_X, x, y,
                    out_eval_tensor.detach().cpu().squeeze(),
                    x_phys_eval.clone().detach().cpu().numpy(),
                    lt, batch_num, i, j, iteration, folder, data_type, sub_sampling
                )
            except Exception:
                pass

        # store series
        print(f"Iter {iteration:04d} | loss={loss_total.item():.4e} | inv_mse={inv_mse.item():.4e} | inv_relL2={inversion_MSE.item():.4e} "
              f"raw[min,max]=({float(x_free.min()):.2f},{float(x_free.max()):.2f}) "
              f"phys[min,max]=({float(x_phys_eval.min()):.2f},{float(x_phys_eval.max()):.2f}) "
              f"||g||={float(g.norm()):.3e}")

        posterior_set.append(x_phys_eval.clone().detach().cpu().numpy())
        out_eval_tensor = out_eval if isinstance(out_eval, torch.Tensor) else torch.tensor(out_eval)
        output_set.append(out_eval_tensor.clone().detach().cpu().numpy())
        gradient_set.append(g.detach().cpu().squeeze())

    return (posterior_set, out_eval_tensor, loss_data_iter, i, j, curves, folder, output_set, gradient_set)

