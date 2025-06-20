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


def plot_observed_only_with_scatter(data, x_idx, y_idx, ax, cmap='viridis'):
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
        r'Ground Truth Output ($u$)',
        r'Ground Truth Input ($a^\ast$)',
        r'Forward Prediction ($\hat{u}$)',
        r'Inversion Result ($a$)',
        r'$|a - a^\ast|$'
    ]

    # your observed locations
    # if sub_sampling == False:
    #     x_idx = cols[:,0].long().cpu().numpy()
    #     y_idx = cols[:,1].long().cpu().numpy()
    # else:
    x_idx = x_idx.detach().cpu().numpy()
    y_idx = y_idx.detach().cpu().numpy()

    fig, axes = plt.subplots(3, 2, figsize=(10,15))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        data = fields[i]
        # choose norm
        # vmin, vmax = data.min(), data.max()
        vmin = np.percentile(data, 0.01)
        vmax = np.percentile(data, 99.99)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        if i in (1, 3):  # only observed points
            if sub_sampling == True:
                sc = ax.scatter(x_idx, y_idx, c=data[y_idx, x_idx], cmap='viridis', norm=norm, s=10, marker='o')
            else:
                sc = ax.scatter(x_idx, y_idx, c=data[y_idx, x_idx], cmap='viridis', norm=norm, s=5, marker='o')
            mappable = sc
            # set the axes limits to match the image‐grid
            ax.set_xlim(-0.5, data.shape[1]-0.5)
            ax.set_ylim(data.shape[0]-0.5, -0.5)      # flip y so origin matches imshow
            ax.set_aspect('equal')

        else:  # full‐field image
            # vmin = -0.49
            # vmax = 0.49
            vmin = np.percentile(data, 0.01)
            vmax = np.percentile(data, 99.99)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
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
        plt.savefig(f"inversion_result_{loss_type}_{num_vec}_{initial_guess}_NGD/inversion_result_{loss_type}_{index}_{iter}.png")
    elif loss_type == "JAC" and top_subsampling == True:
        plt.savefig(f"inversion_result_{loss_type}_{num_vec}_{initial_guess}_top_NGD/inversion_result_{loss_type}_{index}_{iter}.png")
    elif loss_type != "JAC" and top_subsampling == False:
        plt.savefig(f"inversion_result_{loss_type}_{initial_guess}_NGD/inversion_result_{loss_type}_{index}_{iter}.png")
    else:
        plt.savefig(f"inversion_result_{loss_type}_{initial_guess}_top_NGD/inversion_result_{loss_type}_{index}_{iter}.png")
    plt.close(fig)

def project_to_2d_plane(posterior_set, center=None):
    X = np.stack([p.flatten() for p in posterior_set])  # shape: [T, d]
    X = X - X.mean(axis=0) if center is None else X - center.flatten()

    # PCA or manual basis
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    v1, v2 = vh[:2]  # top 2 principal directions

    coords = np.stack([X @ v1, X @ v2], axis=1)  # shape: [T, 2]
    return coords, v1, v2

def evaluate_loss_surface(model, x0_ref, v1, v2, loss_fn, y_true, i, j, steps=21, span=2.0):
    grid_vals = np.linspace(-span, span, steps)
    loss_surface = np.zeros((steps, steps))
    for i1, a in enumerate(grid_vals):
        for i2, b in enumerate(grid_vals):
            x_try = x0_ref.flatten() + a * v1 + b * v2
            x_try_tensor = torch.tensor(x_try.reshape_as(x0_ref), dtype=torch.float32).to(x0_ref.device).requires_grad_(False)
            pred = model(x_try_tensor)
            loss_val = loss_fn(pred[:, :, i, j], y_true[:, :, i, j]).item()
            loss_surface[i1, i2] = loss_val
    return grid_vals, loss_surface

def plot_loss_landscape_with_trajectory(grid_vals, loss_surface, coords, out_path):
    plt.figure(figsize=(8, 6))
    X, Y = np.meshgrid(grid_vals, grid_vals)
    plt.contourf(X, Y, loss_surface, levels=50, cmap='viridis')
    coords = np.array(coords)
    plt.plot(coords[:, 0], coords[:, 1], color='red', marker='o', markersize=2, label='Gradient Descent Path')
    plt.xlabel("Direction 1")
    plt.ylabel("Direction 2")
    plt.title("Loss Landscape and Optimization Trajectory")
    plt.legend()
    plt.colorbar(label="Loss")
    plt.savefig(out_path, dpi=150)
    plt.close()


# def fisher_vec_prod(v, x0, sigma2):
#     v = v.detach().requires_grad_(True)
#     if loss_type == "Devito":
#         x0_np = x0.detach().cpu().numpy()
#         p_fwd = groundwater_model.eval_fwd_op(forcing_term, x0_np, return_array=False)[..., i, j]
#     # ---- Step 1: Compute Jv ---- #
#     if loss_type == "Devito":
#         jvp_out = groundwater_model.compute_linearization(forcing_term.detach().cpu(), x0_np, v.detach().cpu()) # this one does not have observation operator
#         jvp_out = torch.tensor(jvp_out)
#     else:
#         jvp_out = torch.autograd.functional.jvp(lambda x: model(x)[..., i, j].flatten(), (x0,), (v,), create_graph=True)[1]

#     # ---- Step 2: Scale by observation precision matrix: (1/σ²) ---- #
#     weighted = (jvp_out / sigma2).flatten()
    
#     # ---- Step 3: compute JT (1/σ² Jv) ---- #
#     if loss_type == "Devito":
#         print("jvp out", jvp_out.shape, jvp_out.flatten().shape)
#         print("weighted", weighted)
#         probing_vec = weighted @ jvp_out.flatten()
#         print("probing vec", probing_vec, probing_vec.shape)
#         grad = groundwater_model.compute_gradient(x0_np, probing_vec.detach().numpy(), p_fwd)
#         fisher_product = torch.tensor(grad, device=x0.device)
#     else:
#         # print("jvp out", jvp_out.shape, jvp_out.flatten().shape)
#         # print("weighted", weighted)
#         probing_vec = weighted @ jvp_out.flatten()
#         # print("probing vec", probing_vec.shape)

#         fisher_product = torch.autograd.grad((weighted @ jvp_out.flatten()), x0, retain_graph=True)[0]
#     return fisher_product.detach()

# def randomized_fisher_approx(fvp_fn, x0, sigma2, rank=200, n_iter=2):
#     n = x0.numel()
#     device = x0.device
#     Omega = torch.randn(n, rank, device=device)
#     Y = Omega

#     for _ in range(n_iter):
#         Z = []
#         for i in range(rank):
#             v = Y[:, i].reshape_as(x0)
#             Fv = fvp_fn(v, x0, sigma2)
#             Z.append(Fv.reshape(-1))
#         Y = torch.stack(Z, dim=1)

#     Q, _ = torch.linalg.qr(Y)

#     B = torch.zeros(rank, rank, device=device)
#     for i in range(rank):
#         vi = Q[:, i].reshape_as(x0)
#         Fvi = fvp_fn(vi, x0, sigma2).reshape(-1)
#         for j in range(i, rank):
#             vj = Q[:, j].reshape(-1)
#             B[i, j] = torch.dot(Fvi, vj)
#             if i != j:
#                 B[j, i] = B[i, j]

#     return Q, B

def clamp_boundary(x, min_val=-0.5, max_val=0.5):
    with torch.no_grad():
        # bottom_row = x[:, :, -1, :]  # shape: [B, C, W]
        x = torch.clamp(x, min=min_val, max=max_val)
    return x

def fisher_approx_vjp_func(model, x0, i, j, sigma, rank=100):
    """
    Approximate Fisher Information Matrix using orthonormal probing vectors and torch.func.vjp.

    Args:
        model: Callable f(x) that returns shape [1, 1, H, W]
        x0: Tensor with shape [1, 1, H, W] and requires_grad=True
        i, j: Tensors of shape [m], indicating observed locations
        sigma: observation noise std
        rank: number of probing vectors

    Returns:
        Q: Tensor of shape [p, r] such that F ≈ (1/σ²) Q Qᵀ
    """
    device = x0.device
    p = x0.numel()
    m = i.numel()
    rank = min(num_vec, m)  # ← will prevent index error

    # Step 1: Generate orthonormal probing vectors V ∈ R^{m × r}
    V = (1 / sigma) * torch.randn(m, rank, device=device) # normalize
    V, _ = torch.linalg.qr(V)  # Orthonormalize

    # Step 2: Define function from parameters to m-dimensional observed output
    def obs_model(x_flat):
        x = x_flat.view_as(x0)
        if loss_type == "Devito":
            out = model(x, f)
        else:
            out = model(x)  # [1, 1, H, W]
        return out[0, 0, i, j]  # shape: [m]

    if loss_type == "Devito":
        x_flat = x0.detach().clone().requires_grad_(True)                         
    else:
        x_flat = x0.detach().clone().requires_grad_(True).flatten()
    # Step 3: Compute VJPs for each v_k ∈ R^m
    Q_list = []
    for k in range(rank):
        v_k = V[:, k]
        if loss_type == "Devito":
            p_fwd = groundwater_model.eval_fwd_op(forcing_term, x_flat.detach().cpu(), return_array=False) #[..., i, j]
            probe_np = np.zeros((128,128))
            probe_np[i, j] = v_k.detach().cpu().numpy()  # inject probe direction only at observed points
            probe_np = v_k.detach().cpu().numpy()  # inject probe direction only at observed points
            grad = groundwater_model.compute_gradient(
                        x_flat.detach().cpu(), probe_np, p_fwd
                    )                                     # [d,d]
            q_k = torch.tensor(grad, device=x.device).reshape(-1)
        else:
            _, pullback = torch.func.vjp(obs_model, x_flat)
            q_k = pullback(v_k)[0]  # tuple output → take [0]; shape: [p]
        Q_list.append(q_k)

    Q = torch.stack(Q_list, dim=1)  # shape [p, r]
    print("Q shape:", Q.shape)  # Should be [p, rank], e.g. [16384, 100]

    return Q


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

            for k in range(end - start):
                # build 2D probe
                probe_2d = np.zeros_like(x_np, dtype=np.float32)
                flat_vec = V_chunk[:, k].detach().cpu().numpy()
                probe_2d[i.cpu().numpy(), j.cpu().numpy()] = flat_vec

                # compute Jᵀv via Devito
                g2d = model_or_simulator.compute_gradient(x_np, probe_2d, p_fwd)
                grads.append(torch.from_numpy(g2d.reshape(-1)).to(device))

            # stack into [p, chunk_len]
            Q[:, start:end] = torch.stack(grads, dim=1)

            del grads
            torch.cuda.empty_cache()

    return Q

def least_squares_posterior_estimation_fisher(model, input_data, true_data, learning_rate,
                                       batch_num, num_iterations=500, prior=None, i=None, j=None):
    if loss_type != "Devito":
        model.eval()
    mse_loss = torch.nn.MSELoss()

    x0 = input_data.clone().detach().requires_grad_(True).to(device)
    posterior_set = []
    losses, inversion_MSEs, regs, ssims, infty_norm = [], [], [], [], []
    loss_data_iter = []
    start_time = time.time()
    sigma2 = noise_std ** 2  # from your config, e.g., noise_std = 0.2

    for iteration in range(num_iterations):
        x0.grad = None


        ##################
        # Before linesearch
        ##################

        if loss_type == "Devito":
            squeezed_x0 = x0.squeeze()
            squeezed_x0.retain_grad()
            output = model(squeezed_x0)
        else:
            output = model(x0)

        if loss_type == "Devito":
            extracted_output = output[i, j]
        else:
            extracted_output = output[:, :, i, j]
        extracted_target = true_data[:, :, i, j]

        loss = mse_loss(extracted_output.squeeze(), extracted_target.squeeze())
        # reg = mse_loss(x0.squeeze(), prior.squeeze())
        # reg = torch.norm(x0.flatten(), p=2)**2
        reg = torch.zeros(1).to(device)
        # print("reg", reg)
        loss_total = loss + alpha * reg
        loss_total.backward()

        ########################
        # Option 2: Low-rank ver
        ########################
        print("x0 shape:", x0.shape)

        with torch.no_grad():
            if loss_type == "Devito":
                Q = fisher_approx_vjp_batched(groundwater_model, x0, i, j, noise_std,rank=250,chunk_size=50,loss_type=loss_type,forcing_term=forcing_term)
            else:
                Q = fisher_approx_vjp_batched(model, x0, i, j, noise_std, rank=250, chunk_size=100)#125

        # Precondition gradient
        g = x0.grad.detach().flatten()
        B = Q.T @ Q  # [r × r]
        natural_grad = Q @ torch.linalg.solve(B, Q.T @ g)

        lambda_mix = 1 #0.7 #max(0.0, 1.0 - iteration / num_iterations)
        mixed_grad = (1 - lambda_mix) * g.reshape_as(x0) + lambda_mix * natural_grad.reshape_as(x0)

        def loss_fn(x):
            """
            x: torch.Tensor shaped like x0 (e.g. [1,1,H,W]) — assumed on the correct device.
            Returns: scalar float = MSE(observed) + α·gradient_penalty(x)
            """
            with torch.no_grad():
                # 1) forward
                target = true_data[0, 0, i, j]
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
                data_misfit = mse_loss(pred, target)

                # 4) combine and return as Python float
                return float(data_misfit.item())

        if iteration % decay_interval == 0 and iteration > 0:
            # re-tune the step once in a cheap way
            tuned_lr = backtracking_step(
                x0, loss_fn, mixed_grad, learning_rate,
                c=1e-4, beta=0.7, max_ls_iters=2
            )
            learning_rate = tuned_lr  # update your base for the next block
            print("new lr", tuned_lr)
        # just a fixed step
        with torch.no_grad():
            x0 -= learning_rate * mixed_grad
            x0.requires_grad_(True)

        # # ---------- NEW: hybrid NGD + GD direction ----------
        # g  = x0.grad.detach().flatten()          # [Npar]

        # # 1) projection onto the retained sub-space
        # coeff     = Q.T @ g                      # [r]   = Qᵀ g
        # B         = Q.T @ Q                      # [r×r]  (small!)
        # damping   = 1e-7                         # tweak if needed
        # B_chol    = torch.linalg.cholesky(B + damping*torch.eye(B.size(0), device=B.device))
        # d_in      = Q @ torch.cholesky_solve(coeff.unsqueeze(1), B_chol)[:,0]   # NGD step in sub-space

        # # 2) complement (no curvature info)
        # g_in      = Q @ coeff                    # = projection of g
        # g_out     = g - g_in                     # = (I − QQᵀ) g
        # print("‖g_in‖₂ =", g_in.norm().item(),"‖g_out‖₂ =", g_out.norm().item())


        # # 3) hybrid update
        # η     = learning_rate        # step in well-determined directions
        # β     = 0.0001 * learning_rate  # step in null-space  (tune!)
        # step  = -η * d_in - β * g_out
        # print(torch.norm(d_in), torch.norm(g_out))

        # # ---------- (optional) very cheap back-tracking ----------
        # if iteration % decay_interval == 0 and iteration > 0:
        #     tuned_lr = backtracking_step(
        #         x0, loss_fn, step.reshape_as(x0), 1.0,
        #         c=1e-4, beta=0.7, max_ls_iters=2
        #     )
        #     step *= tuned_lr

        # # ---------- apply update ----------
        # with torch.no_grad():
        #     x0 += step.reshape_as(x0)
        #     x0.requires_grad_(True)


        # print(f"Step norm: {η.norm():.4e}")

        diff = x0 - prior
        inversion_MSE = torch.norm(diff) / torch.norm(prior)
        input_numpy = x0.detach().cpu().squeeze().numpy()
        prior_numpy = prior.detach().cpu().squeeze().numpy()
        ssim_value = ssim(input_numpy.astype(np.float64),
                          prior_numpy.astype(np.float64),
                          data_range=float(input_numpy.max() - input_numpy.min()))

        elapsed = time.time() - start_time

        loss_data_iter.append({
            "sample":        batch_num,
            "iteration":     iteration,
            "elapsed_s":     elapsed,
            "loss":          loss_total.item(),
            "inversion_MSE": inversion_MSE.item(),
            "regularization":reg.item(),
            "SSIM":          ssim_value
        })

        if batch_num < 2 and iteration % 1 == 0:
            gradient = g.cpu().squeeze().reshape(128,128)
            plt.imshow(gradient.numpy(), cmap='viridis')
            plt.colorbar(label='Gradient Value', shrink=0.8)
            plt.title('Gradient w.r.t. Input x0')
            if loss_type == "JAC":
                plt.savefig(f'inversion_result_{loss_type}_{num_vec}_{initial_guess}_NGD/iter={batch_num}_gradient_{iteration}.png')
                plot_single(x0.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{num_vec}_{initial_guess}_NGD/iter={batch_num}_inversion_{iteration}.png')
                plot_single(output.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{num_vec}_{initial_guess}_NGD/iter={batch_num}_inversion_{iteration}_output.png')
                plot_inversion_result(zero_X, x, y, output.detach().cpu().squeeze(), x0.clone().detach().cpu(), loss_type, batch_num, i, j, iteration)
            elif top_subsampling:
                plt.savefig(f'inversion_result_{loss_type}_{initial_guess}_top_NGD/iter={batch_num}_gradient_{iteration}.png')
                plot_single(x0.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{initial_guess}_top_NGD/iter={batch_num}_inversion_{iteration}.png')
                plot_single(output.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{initial_guess}_top_NGD/iter={batch_num}_inversion_{iteration}_output.png')
                plot_inversion_result(zero_X, x, y, output.detach().cpu().squeeze(), x0.clone().detach().cpu(), loss_type, batch_num, i, j, iteration)
            else:
                plt.savefig(f'inversion_result_{loss_type}_{initial_guess}_NGD/iter={batch_num}_gradient_{iteration}.png')
                plot_single(x0.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{initial_guess}_NGD/iter={batch_num}_inversion_{iteration}.png')
                plot_single(output.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{initial_guess}_NGD/iter={batch_num}_inversion_{iteration}_output.png')
                plot_inversion_result(zero_X, x, y, output.detach().cpu().squeeze(), x0.clone().detach().cpu(), loss_type, batch_num, i, j, iteration)

        print(f"Iteration {iteration}, Loss: {loss_total.item():.4e}", inversion_MSE.item(), ssim_value, flush=True)

        losses.append(loss_total.item())
        inversion_MSEs.append(inversion_MSE.item())
        regs.append(reg.item())
        ssims.append(ssim_value)
        posterior_set.append(x0.clone().detach().cpu().numpy())

    return posterior_set, losses, inversion_MSEs, regs, ssims, output.detach().cpu().squeeze(), loss_data_iter, i, j

# ----------------------
# Main Script for Inversion on Multiple Samples (batch_size=1)
# ----------------------
if __name__ == "__main__":
    # Set up device and random seed.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print(f"Using device: {device}")

    # Define simulation parameters.
    num_vec = 200
    loss_type = "MSE"  # "JAC" "MSE" "Devito"
    alpha = 0.4 #1e-6 #0.05
    noise_std = 1.0 #0.3
    initial_guess = "smooth" # "smooth", "noisy"
    sub_sampling = True
    top_subsampling = False
    full_obs = False
    decay_interval = 1
    # damping_lambda = 0 #5e-7

    if initial_guess == "prior_mean":
        learning_rate = 10 # 0.0001 (grf, fullobs) #0.005 (noisy, fullobs) #0.00005  # Inversion learning rate.
        num_sample = 1 #1
        num_sample_prior = 100 #5
        num_epoch = 2001 #1001
        offset=128
    elif initial_guess == "smooth":
        learning_rate = 10 #0.5 #100 #0.0001 # 0.0001 (grf, fullobs) #0.005 (noisy, fullobs) #0.00005  # Inversion learning rate.
        num_sample = 1 #3 #50
        num_sample_prior = 100
        num_epoch = 2000 #35001 #500 #2201 #2001 #1001
        offset=128
        GRF = 3
        if GRF == 1:
            kernel_size = 9 #55 #(grf, fullobs)
            sigma = 2.0 #100.0 # (grf, fullobs)
        elif GRF == 2:
            kernel_size = 55
            sigma = 100.0
        elif GRF == 3:
            kernel_size = 19
            sigma = 300.0
    
    
    # Load configuration and dataset. and checkpoint
    if loss_type == "JAC" and num_vec == 1:
        config = "configs/eigenvectors/e=1.yaml"
        ckpt_path = "checkpoints/n=128_e=1_m=FNO_s=RFS_l=JAC_20250513_164312/n=128_e=1_m=FNO_s=RFS_l=JAC_epoch=249_val_rel_l2_loss=0.0006.ckpt"
    elif loss_type == "JAC" and num_vec == 10:
        config = "output/n=128_e=10_m=FNO_s=RFS_l=JAC_20250512_144619/config.yaml"
        ckpt_path = "checkpoints/n=128_e=10_m=FNO_s=RFS_l=JAC_20250512_144619/n=128_e=10_m=FNO_s=RFS_l=JAC_epoch=249_val_rel_l2_loss=0.0005.ckpt"
    elif loss_type == "JAC" and num_vec == 50:
        config = load_config("output/n=128_e=50_m=FNO_s=RFS_l=JAC_20250512_141821/config.yaml")
        ckpt_path = f"checkpoints/n=128_e=50_m=FNO_s=RFS_l=JAC_20250514_151731/n=128_e=50_m=FNO_s=RFS_l=JAC_epoch=249_val_rel_l2_loss=0.0004.ckpt"
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
    elif loss_type == "RAND":
        config = load_config("output/n=128_e=8_m=FNO_s=RAND_l=JAC_20250421_124311/config.yaml")
        ckpt_path = f"checkpoints/n=128_e=8_m=FNO_s=RAND_l=JAC_20250421_125959/last.ckpt"
    elif loss_type == "MSE":
        config = load_config("configs/darcy_MSE.yaml")
        ckpt_path = "checkpoints/DARCY_MSE/Darcy_training_epoch=249_val_rel_l2_loss=0.0009_MSE_May14.ckpt"
    
    if loss_type != "Devito":
        model = NSModel.load_from_checkpoint(ckpt_path).eval().to(device)
        with open("rng_state_devito.pkl", "rb") as f:
            state = pickle.load(f)
            np.random.set_state(state["np_random_state"])
            random.setstate(state["random_state"])

    # Load Data
    if num_vec == 200:
        print("200!")
        data_config = load_config("output/n=400_e=200_m=FNO_s=RFS_l=JAC_20250615_133916/config.yaml")
        print(data_config.experiment.dataset_type, data_config.data_settings)
        dataset = get_dataset(data_config.experiment.dataset_type, data_config.data_settings)
        dataloader = dataset.get_dataloader(offset=414, limit=num_sample)
        prior_dataloader = dataset.get_dataloader(offset=410, limit=num_sample_prior)
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

    print("saving h5 files...")
    if loss_type == "JAC" and top_subsampling == False :
        fname = f'inversion_history_{loss_type}_{num_vec}_{initial_guess}_NGD.h5'
    elif loss_type == "JAC" and top_subsampling == True:
        fname = f'inversion_history_{loss_type}_{num_vec}_{initial_guess}_top_NGD.h5'
    elif loss_type != "JAC" and top_subsampling == False :
        fname = f'inversion_history_{loss_type}_{initial_guess}_NGD.h5'
    else:
        fname = f'inversion_history_{loss_type}_{initial_guess}_top_NGD.h5'
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
        sum_x = 0.0
        n_samples = 0
        for batch in prior_dataloader:
            for sample_num, x in enumerate(batch['x']):
                x = x.to(device)
                sum_x += x.squeeze()
                n_samples += 1

        print("Prior averaged over ", n_samples, sum_x.shape)
        prior_mean = sum_x / n_samples  # shape: [C, H, W]
        prior_mean = prior_mean.unsqueeze(dim=0).unsqueeze(dim=1).detach()


    # Prepare CSV accumulators:
    loss_data_all = []
    sample_counter = 0

    for batch in dataloader:
        x = batch['x'].to(device)
        y = batch['y'].to(device) + torch.randn_like(x) * noise_std
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
            subsample_ratio = 1#0.15
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
            print("i", i.shape, j.shape)

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
            # zero_X = apply_gaussian_smoothing(x, kernel_size, sigma)
            zero_X = torch.ones(x.shape).to(device) * x.amin()
            zero_X[..., 45:-45] = x.amax()
            zero_X = zero_X.detach()
        elif initial_guess == "noisy":
            zero_X = x + torch.randn_like(x) * noise_std
        elif initial_guess == "prior_mean":
            zero_X = prior_mean
        plot_single(zero_X.detach().cpu().squeeze(), f"zero_X_sample_{sample_counter}.png", "jet")

        forcing_term = torch.zeros(zero_X.squeeze().shape)
        groundwater_model = GroundwaterEquation(forcing_term.shape[0])
        if loss_type == "Devito":
            gw_torch_model = GroundwaterModel(forcing_term.shape[0])
            model = lambda x: gw_torch_model(x, forcing_term)


        posterior_set, losses, inversion_MSEs, regs, ssims, pred, loss_data_iter, i_idx, j_idx = (
            least_squares_posterior_estimation_fisher(
                model, zero_X, y,
                learning_rate, batch_num=sample_counter,
                num_iterations=num_epoch, prior=x, i=i, j=j
            )
        )

        

        # Plot the final inversion result.
        final_x0 = torch.tensor(posterior_set[-1]).detach()
        plot_inversion_result(zero_X, x, y, pred, final_x0, loss_type, sample_counter, i_idx, j_idx, num_epoch)


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

    # # Convert posterior_set to 2D projection
    # coords_2d, v1, v2 = project_to_2d_plane(posterior_set)
    # # Reference point (e.g., initial guess or center)
    # x0_ref = torch.tensor(posterior_set[0], dtype=torch.float32).to(device)
    # # Evaluate loss surface
    # grid_vals, loss_surface = evaluate_loss_surface(model, x0_ref, v1, v2, mse_loss, y, i_idx, j_idx)
    # # Plot
    # plot_loss_landscape_with_trajectory(grid_vals, loss_surface, coords_2d, f"loss_landscape_trajectory_{sample_counter}.png")