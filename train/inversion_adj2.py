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
        x_pred,    # inversion result
        np.abs(x.squeeze().cpu().numpy() - x_pred)
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
                sc = ax.scatter(x_idx, y_idx, c=data[y_idx, x_idx], cmap='jet', norm=norm, s=10, marker='o')
            else:
                sc = ax.scatter(x_idx, y_idx, c=data[y_idx, x_idx], cmap='jet', norm=norm, s=5, marker='o')
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

def clamp_boundary(x, min_val=-0.5, max_val=0.5):
    with torch.no_grad():
        # bottom_row = x[:, :, -1, :]  # shape: [B, C, W]
        x = torch.clamp(x, min=min_val, max=max_val)
    return x

def fisher_approx_vjp_func(model, x0, i, j, sigma, rank=100, groundwater_model=None, forcing_term=None):
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
    # rank = min(rank, m)
    Q_list = []   

    # Step 1: Generate orthonormal probing vectors V ∈ R^{m × r}
    V_raw = torch.randn(m, rank, device=device)
    V, _ = torch.linalg.qr(V_raw)
    # V = V / sigma

    # Step 2: Define function from parameters to m-dimensional observed output
    def obs_model(x_flat):
        x = x_flat.view_as(x0)
        if loss_type == "Devito":
            out = model(x, f)
        else:
            out = model(x)  # [1, 1, H, W]
        return out[0, 0, i, j]  # shape: [m] 

    # Step 3: Compute VJPs for each v_k ∈ R^m
    for k in range(rank):
        v_k = V[:, k]
        x_probe = x0.detach().clone().requires_grad_(True)
        
        if loss_type == "Devito":
            p_fwd = groundwater_model.eval_fwd_op(forcing_term, x_probe.detach().cpu(), return_array=False) #[..., i, j]
            probe_np = np.zeros((128,128))
            probe_np[i.cpu().numpy(), j.cpu().numpy()] = v_k.detach().cpu().numpy()
            grad = groundwater_model.compute_gradient(
                        x_probe.detach().cpu(), probe_np, p_fwd
                    )                                     # [d,d]
            q_k = torch.tensor(grad, device=x_probe.device).reshape(-1)
            print(f"‖q_k‖ = {q_k.norm().item()}")
        else:
            _, pullback = torch.func.vjp(obs_model, x_probe.flatten())
            q_k = pullback(v_k)[0]  # tuple output → take [0]; shape: [p]
            print(f"‖q_k‖ = {q_k.norm().item()}")
        Q_list.append(q_k)

    Q = torch.stack(Q_list, dim=1)  # shape [p, r]
    print("Q shape:", Q.shape)  # Should be [p, rank], e.g. [16384, 100]

    return Q


def least_squares_posterior_estimation_fisher(model, input_data, true_data, learning_rate, reg,
                                       batch_num, groundwater_model, forcing_term, num_iterations=500, prior=None, i=None, j=None):
    if loss_type != "Devito":
        model.eval()
    mse_loss = torch.nn.MSELoss()

    x0 = input_data.clone().detach().requires_grad_(True).to(device)
    posterior_set = []
    losses, inversion_MSEs, regs, ssims, infty_norm = [], [], [], [], []
    loss_data_iter = []
    start_time = time.time()
    sigma2 = noise_std ** 2  # from your config, e.g., noise_std = 0.2
    lambda_mix = 1.0

    for iteration in range(num_iterations):
        x0.grad = None

        if loss_type == "Devito":
            squeezed_x0 = x0.squeeze()
            squeezed_x0.retain_grad()
            output = model(squeezed_x0)
        else:
            output = model(x0)

        if loss_type == "Devito":
            extracted_output = output[i, j]
            extracted_target = true_data[:, :, i, j]
        else:
            extracted_output = output[:, :, i, j]
            extracted_target = true_data[:, :, i, j]

        # loss = mse_loss(extracted_output.squeeze(), extracted_target.squeeze())
        # if reg == True:
        #     reg = gradient_penalty(x0)
        #     # reg = torch.norm(x0.flatten(), p=2)**2
        # else:
        #     reg = torch.tensor(0)
        # loss_total = loss + alpha * reg


        #######################
        # Add 
        #######################

        # 1) CG solver (no graph, small damping for stability)
        def cg(A_mv, b, tol=1e-6, maxiter=20):
            x = torch.zeros_like(b)
            r = b - A_mv(x)
            p = r.clone()
            rs = torch.dot(r,r)
            for _ in range(maxiter):
                Ap = A_mv(p)
                α  = rs / (torch.dot(p, Ap) + 1e-12)
                x += α*p
                r -= α*Ap
                rs_new = torch.dot(r,r)
                if torch.sqrt(rs_new) < tol: break
                p = r + (rs_new/rs)*p
                rs = rs_new
            return x

        # 2) Build your matvec for (JᵀJ + λI)
        def make_matvec(model, x0, i, j, σ2=1.0, λ=1e-3):
            x_flat0 = x0.detach().clone().requires_grad_(True).flatten()
            def res(xf):
                x = xf.view_as(x0)
                pred = model(x)[0,0,i,j]
                return (pred - extracted_target) / σ2   # residual vector ∈ ℝᵐ

            def A_mv(v):
                # Jv via forward‐mode
                _, jv = torch.autograd.functional.jvp(res, (x_flat0,), (v,), create_graph=False)
                jv = jv.detach()
                # Jᵀ (Jv) via VJP
                _, pull = torch.func.vjp(res, x_flat0)
                jTJv, = pull(jv)
                return jTJv.detach() + λ*v

            return A_mv, x_flat0

        # 3) One NG step in your loop:
        A_mv, x_flat0 = make_matvec(model, x0, i, j, σ2=noise_std**2, λ=0.)

        # compute gradient g = ∇½‖r‖²
        def loss_fn(xf):
            r = ((model(xf.view_as(x0))[0,0,i,j] - extracted_target) / (noise_std**2)).squeeze()
            print(r.shape)
            return 0.5*torch.dot(r,r)
        loss_total = loss_fn(x_flat0)
        g = torch.autograd.grad(loss_total, x_flat0)[0].detach()

        # solve (JᵀJ + λI) η = −g
        with torch.no_grad():
            η = cg(A_mv, -g, tol=1e-5, maxiter=20).view_as(x0)
            print("step‐norm =", η.norm().item())     # should now be nonzero
            x0 -= learning_rate * η
            x0.requires_grad_(True)

        ########################
        # Option 2: Low-rank ver
        ########################
        # # 1) forward & loss under no_grad if possible OR with grads if reg needs it
        # x0.grad = None
        # loss_total.backward()               # only once per iteration
        # g = x0.grad.detach().flatten()
        # print("‖g‖ =", g.norm().item())
        # x0.grad.zero_()

        # # 2) VJP on a fresh graph
        # x0_probe = x0.detach().clone().requires_grad_(True)
        # if loss_type == "Devito":
        #     Q = fisher_approx_vjp_func(gw_torch_model, x0_probe, i, j, sigma=noise_std, rank=500, groundwater_model= groundwater_model, forcing_term=forcing_term)
        # else:
        #     Q = fisher_approx_vjp_func(model, x0_probe, i, j, sigma=noise_std, rank=150)

        # # 3) natural-gradient update under no_grad
        # with torch.no_grad():
        #     coeffs = Q.T @ g           # shape [r]
        #     print("Q^T g:", coeffs)
        #     B = Q.T @ Q
        #     nat_grad = Q @ torch.linalg.solve(B, Q.T @ g)
        #     print("raw step‐norm =", nat_grad.norm().item())
        #     x0 -= learning_rate * ((1-lambda_mix)*g.reshape_as(x0) + lambda_mix*nat_grad.reshape_as(x0))
        #     x0.requires_grad_(True)
        #     # x0.data = torch.clamp(x0.data, min= prior.min() * 1.3, max= prior.max() * 1.3)
        
        # 5. Stream snapshot to HDF5
        snapshot = x0.detach().cpu().numpy().squeeze()
        dset[batch_num, iteration, :, :] = snapshot

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
            # "regularization":reg.item(),
            "SSIM":          ssim_value
        })

        if batch_num < 2 and iteration % 1 == 0:
            gradient = g.cpu().squeeze().reshape(128,128)
            plt.imshow(gradient.numpy(), cmap='viridis')
            plt.colorbar(label='Gradient Value', shrink=0.8)
            plt.title('Gradient w.r.t. Input x0')
            if loss_type == "JAC":
                plt.savefig(f'inversion_result_{loss_type}_{num_vec}_{initial_guess}_NGD/iter={batch_num}_gradient_{iteration}.png')
                plot_single(snapshot, f'inversion_result_{loss_type}_{num_vec}_{initial_guess}_NGD/iter={batch_num}_inversion_{iteration}.png')
                # plot_single(output.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{num_vec}_{initial_guess}_NGD/iter={batch_num}_inversion_{iteration}_output.png')
                plot_inversion_result(zero_X, x, y, output.detach().cpu().squeeze(), snapshot, loss_type, batch_num, i, j, iteration)
            elif top_subsampling:
                plt.savefig(f'inversion_result_{loss_type}_{initial_guess}_top_NGD/iter={batch_num}_gradient_{iteration}.png')
                plot_single(snapshot, f'inversion_result_{loss_type}_{initial_guess}_top_NGD/iter={batch_num}_inversion_{iteration}.png')
                # plot_single(output.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{initial_guess}_top_NGD/iter={batch_num}_inversion_{iteration}_output.png')
                plot_inversion_result(zero_X, x, y, output.detach().cpu().squeeze(), snapshot, loss_type, batch_num, i, j, iteration)
            else:
                plt.savefig(f'inversion_result_{loss_type}_{initial_guess}_NGD/iter={batch_num}_gradient_{iteration}.png')
                plot_single(snapshot, f'inversion_result_{loss_type}_{initial_guess}_NGD/iter={batch_num}_inversion_{iteration}.png')
                # plot_single(output.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{initial_guess}_NGD/iter={batch_num}_inversion_{iteration}_output.png')
                plot_inversion_result(zero_X, x, y, output.detach().cpu().squeeze(), snapshot, loss_type, batch_num, i, j, iteration)

        print(f"Iteration {iteration}, Loss: {loss_total.item():.4e}", inversion_MSE.item(), ssim_value, flush=True)

        losses.append(loss_total.item())
        inversion_MSEs.append(inversion_MSE.item())
        # regs.append(reg.item())
        ssims.append(ssim_value)

    return losses, inversion_MSEs, regs, ssims, output.detach().cpu().squeeze(), loss_data_iter, i, j

# ----------------------
# Main Script for Inversion on Multiple Samples (batch_size=1)
# ----------------------
if __name__ == "__main__":
    # Set up device and random seed.
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print(f"Using device: {device}")

    # Define simulation parameters.
    num_vec = 100
    loss_type = "MSE"  # or "JAC" "MSE" "Devito"
    alpha = 0 #1e-6 #0.05
    noise_std = 1.0 #0.3
    initial_guess = "smooth" # "smooth", "noisy"
    sub_sampling = True
    top_subsampling = False
    full_obs = False
    reg = False
    # damping_lambda = 0 #5e-7

    if initial_guess == "prior_mean":
        learning_rate = 0.001 # 0.0001 (grf, fullobs) #0.005 (noisy, fullobs) #0.00005  # Inversion learning rate.
        num_sample = 10 #1
        num_sample_prior = 100 #5
        num_epoch = 2001 #1001
        offset=130
    elif initial_guess == "smooth":
        learning_rate = 1e-7 #0.5 #100 #0.0001 # 0.0001 (grf, fullobs) #0.005 (noisy, fullobs) #0.00005  # Inversion learning rate.
        num_sample = 1 #3 #50
        num_sample_prior = 100
        num_epoch = 30000 #35001 #500 #2201 #2001 #1001
        offset=128
        GRF = 2
        if GRF == 1:
            kernel_size = 45 #55 #(grf, fullobs)
            sigma = 10.0 #100.0 # (grf, fullobs)
        elif GRF == 2:
            kernel_size = 55
            sigma = 100.0
        elif GRF == 3:
            kernel_size = 55
            sigma = 50.0
    
    
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
            x = batch['x'].to(device)
            sum_x += x.squeeze()
            n_samples += 1

        print("Prior averaged over ", n_samples)
        prior_mean = sum_x / n_samples  # shape: [C, H, W]
        prior_mean = prior_mean.unsqueeze(dim=0).unsqueeze(dim=1).detach()

    # Prepare CSV accumulators:
    loss_data_all = []
    sample_counter = 0

    for batch in dataloader:
        print("batch", batch)
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
            zero_X = apply_gaussian_smoothing(x, kernel_size, sigma) #+ torch.randn_like(x) * 0.1
        elif initial_guess == "noisy":
            zero_X = x + torch.randn_like(x) * noise_std
        elif initial_guess == "prior_mean":
            zero_X = prior_mean
        plot_single(zero_X.detach().cpu().squeeze(), f"zero_X_sample_{sample_counter}.png", "jet")

        if loss_type == "Devito":
            forcing_term = torch.zeros(zero_X.squeeze().shape)
            groundwater_model = GroundwaterEquation(forcing_term.shape[0])
            gw_torch_model = GroundwaterModel(forcing_term.shape[0])
            model = lambda x: gw_torch_model(x, forcing_term)
        else:
            groundwater_model, forcing_term =None, None


        losses, inversion_MSEs, regs, ssims, pred, loss_data_iter, i_idx, j_idx = (
            least_squares_posterior_estimation_fisher(
                model, zero_X, y,
                learning_rate, reg, groundwater_model=groundwater_model, forcing_term=forcing_term, batch_num=sample_counter,
                num_iterations=num_epoch, prior=x, i=i, j=j
            )
        )
        
            

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