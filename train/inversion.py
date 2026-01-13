import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from physicsnemo.models.fno import FNO
import h5py
from skimage.metrics import structural_similarity as ssim
from scipy.interpolate import interp1d
import time
import sys
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
from groundwater.devito_op import GroundwaterModel, GroundwaterLayer, GroundwaterEquation
from groundwater.utils import GaussianRandomField, plot_fields
import h5py
import pickle
import os
import random
from models.ns_inversion import NSModel
from utils import * 
from utils_plot import *
from utils_inversion import *

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datasets.simulators.NS import NavierStokesSimulator

# NUM_PSEUDO_TIMESTEPS: int = 500000

# ----------------------
# Helper Functions
# ----------------------

def compute_gradient_single(simulator, x_np, probe_2d, p_fwd):
    """
    CPU worker: runs one Devito gradient (J^T v) and returns a flat array.
    """
    g2d = simulator.compute_gradient(x_np, probe_2d, p_fwd)  # [H,W]
    return g2d.reshape(-1)  # → (p,)

def fisher_approx_vjp_batched(model_or_simulator, x0, i, j, sigma, rank=200, chunk_size=32, opt_type="FNO", forcing_term=None, 
    noise_std = 0.01, full_obs = True):

    device = x0.device
    p = x0.numel()       # total #parameters = H*W
    m = i.numel()        # #observations

    # 1) sample & orthonormalize all probes: V_full ∈ R^{m×r}
    if noise_std == 0:
        V_full = torch.randn(m, rank, device=device)
    else:
        V_full = (1.0 / sigma) * torch.randn(m, rank, device=device)
    V_full, _ = torch.linalg.qr(V_full)
    print("m", m, "rank", rank)
    print("V", V_full.shape)

    # prepare result buffer
    Q = torch.empty(p, rank, device=device)

    # Devito: do one forward solve
    if opt_type == "Devito":
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

        if opt_type != "Devito":
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

            # probe_2d = np.zeros_like(x_np, dtype=np.float32)
            probe_2d = np.zeros((chunk_size, p))
            # flat_vec = V_chunk[:, k].detach().cpu().numpy()
            flat_vec = V_chunk.detach().cpu().numpy().reshape(chunk_size, -1)    
            print("flatvec", flat_vec.shape)
            probe_flat = np.zeros((chunk_size, p))   

            if full_obs == True:
                probe_2d = flat_vec
            else:
                # idx = torch.tensor(L.detach().cpu(), dtype=torch.long)                
                rows = np.arange(chunk_size).reshape(-1, 1)  
                probe_2d[rows, i] = flat_vec
            print("probe_2d", probe_2d.shape)
            print("i", i)
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


def two_sided_armijo_line_search(x, loss_fn, grad, step0, c=5e-1, beta=0.8, gamma=1.02,
                       max_expand=2, max_shrink=3):
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


# Binary projection from PHYSICAL field (consistent with tanh range)
def binarize_from_phys(x_phys, a_min=0.1, a_max=0.9, mid=None):
    if mid is None:
        mid = 0.5 * (a_min + a_max)  # 0.5 when [0.1,0.9]
    return torch.where(
        x_phys >= mid,
        torch.tensor(a_max, device=x_phys.device, dtype=x_phys.dtype),
        torch.tensor(a_min, device=x_phys.device, dtype=x_phys.dtype),
    )

def darcy_mask_tanh(x):
    a_max = 0.9
    a_min = 0.1
    k = torch.log1p(torch.exp(x))
    # Normalize to range [a_min, a_max]
    k = a_min + (a_max - a_min) * (k / (1 + k))
    # return 0.5 * (a_max - a_min) * torch.tanh(x) + 0.5 * (a_max + a_min)
    return k

def least_squares_posterior_estimation_prev(
    model, input_data, true_data, learning_rate,batch_num, num_iterations=500, prior=None,i=None, j=None, folder=None):

    if opt_type != "Devito" or data_type != "Darcy":
        model.eval()

    # x0 will represent the KL coefficients (not the pixel field)
    ncmp = 8                                   # number of cosine modes per dimension
    x0 = torch.zeros(ncmp**2, requires_grad=True, device=device)
    # x0 = input_data.clone().detach().requires_grad_(True).to(device)
    x00 = x0.clone().detach()
    initial_guess, k_field = reconstruct_fourier_field(x00, dim, dim, ncmp, s=1.0, alpha=1.0, sigma=1.5, device=device)


    posterior_set, output_set, curves = [], [], []
    losses, inversion_MSEs, regs, ssims, infty_norm, gradient_set, loss_data_iter = [], [], [], [], [], [], []
    start_time = time.time()
    true_data = true_data.to(device)
    mse_loss = torch.nn.MSELoss(reduction='mean')

    sigma_y = 1e-2
    optimizer = torch.optim.LBFGS([x0], lr=0.05, max_iter=1, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        # reconstruct log-permeability u(x) from coefficients x0
        # u_field = u_from_theta(x0, ncmp=ncmp, nx=dim, ny=dim, s=1.0, alpha=1.0, sigma=1.5)
        u_field, k_field = reconstruct_fourier_field(x0, dim, dim, ncmp, s=1.0, alpha=1.0, sigma=1.5, device=device)
        # forward PDE solve (Darcy)
        if opt_type == "Devito":
            y_pred = model(u_field)
        else:
            y_pred = model(u_field.unsqueeze(0).unsqueeze(0))
        if full_obs:
            pred = y_pred.flatten()
            target = true_data.flatten()
        else:
            pred = y_pred.squeeze()[i, j]
            target = true_data.squeeze()[i, j]

        loss_data = ((pred - target) / sigma_y).pow(2).sum()
        # loss_prior = 0.5 * x0.pow(2).sum()
        loss_total = loss_data #+ loss_prior
        loss_total.backward()
        return loss_total

    for iteration in range(num_iterations):
        loss_total = optimizer.step(closure)
        # u_field = u_from_theta(x0, ncmp=ncmp, nx=dim, ny=dim, s=1.0, alpha=1.0, sigma=1.5)
        u_field, k_field = reconstruct_fourier_field(x0.clone().detach(), dim, dim, ncmp, s=1.0, alpha=1.0, sigma=1.5, device=device)
        if iteration == 0:
            initial_guess = initial_guess.detach().clone().cpu()
            plot_single(u_field.detach().cpu().squeeze(),
            f'{folder}/iter={batch_num}_inversion_{iteration}_initial_guess.png',
            vmin=prior.min().item(), vmax=prior.max().item(), cmap='BrBG')
            plot_single(prior.detach().cpu().squeeze(),
            f'{folder}/iter={batch_num}_inversion_{iteration}_prior.png',
            vmin=prior.min().item(), vmax=prior.max().item(), cmap='BrBG')
        if opt_type == "Devito":
            output = model(u_field)
        else: 
            output = model(u_field.unsqueeze(0).unsqueeze(0))

        # record gradients (w.r.t. coefficients)
        g = x0.grad.detach().clone() if x0.grad is not None else torch.zeros_like(x0)

        # compute simple diagnostics for logging
        inv_mse = torch.sqrt(torch.mean((u_field - prior.squeeze())**2))
        elapsed = time.time() - start_time

        loss_data_iter.append({
            "sample": batch_num,
            "iteration": iteration,
            "elapsed_s": elapsed,
            "loss": loss_total.item(),
            "inversion_MSE": inv_mse.item()
        })

        print(f"[Iter {iteration}] loss={loss_total.item():.3e}, ||grad||={g.norm().item():.3e}")

        # save visualizations every few iterations
        if iteration % save_fig_iter == 0:
            plot_single(u_field.detach().cpu().squeeze(),
                        f'{folder}/iter={batch_num}_inversion_{iteration}.png',
                        vmin=prior.min().item(), vmax=prior.max().item(), cmap='BrBG')
            plot_single(x0.detach().clone().cpu().squeeze().reshape(8,8), f'{folder}/iter={batch_num}_inversion_{iteration}_coeff.png',
                        vmin=-3.0, vmax=4.0, cmap='plasma')
            plot_single(g.detach().cpu().reshape(8,8).numpy(), f'{folder}/iter={batch_num}_inversion_{iteration}_grad.png', cmap='PRGn')

        posterior_set.append(u_field.detach().cpu().numpy())
        output_set.append(output.detach().cpu().numpy())
        gradient_set.append(g.detach().cpu().reshape(8,8).numpy())

    return posterior_set, output, loss_data_iter, i, j, curves, folder, output_set, gradient_set, initial_guess

# def least_squares_posterior_estimation_prev(
#     model, input_data, true_data, learning_rate,
#     batch_num, num_iterations=500, prior=None,
#     i=None, j=None, folder=None):
#     """
#     Gradient-descent version of least-squares inversion
#     with same plotting and return values as before.
#     """
#     if opt_type != "Devito" or data_type != "Darcy":
#         model.eval()
#     device = input_data.device
#     true_data = true_data.to(device)
#     sigma_y = 1e-2
#     ncmp = 8
#     mse_loss = torch.nn.MSELoss(reduction='mean')

#     # Initialize optimization variable
#     # x0 = input_data.clone().detach().requires_grad_(True)
#     # x0 = torch.ones(ncmp**2, requires_grad=True, device=device)
#     x0 = torch.rand(ncmp**2, requires_grad=True, device=device)
#     optimizer = torch.optim.SGD([x0], lr=learning_rate, momentum=0.0) 

#     # Prepare logging containers
#     posterior_set, output_set, curves = [], [], []
#     losses, inversion_MSEs, regs, ssims, infty_norm, gradient_set, loss_data_iter = [], [], [], [], [], [], []

#     start_time = time.time()
#     print(f"[Start GD inversion] lr={learning_rate}, iterations={num_iterations}")

#     for iteration in range(num_iterations):
#         optimizer.zero_grad()

#         # reconstruct physical field from coefficients (Fourier field)
#         u_field, k_field = reconstruct_fourier_field(
#             x0, dim, dim, ncmp, s=1.0, alpha=1.0, sigma=1.5, device=device
#         )

#         # forward PDE / surrogate solve
#         if opt_type == "Devito":
#             y_pred = model(u_field)
#         else:
#             y_pred = model(u_field.unsqueeze(0).unsqueeze(0))

#         # compute data misfit
#         if full_obs:
#             pred = y_pred.flatten()
#             target = true_data.flatten()
#         else:
#             pred = y_pred.squeeze()[i, j]
#             target = true_data.squeeze()[i, j]

#         loss_total = ((pred - target) / sigma_y).pow(2).mean()
#         loss_total.backward()
#         optimizer.step()

#         # clamp to physical range
#         with torch.no_grad():
#             min_clamp = prior.min().item() - 0.01
#             max_clamp = prior.max().item() + 0.01
#             x0.clamp_(min=min_clamp, max=max_clamp)

#         # record gradient
#         g = x0.grad.detach().clone() if x0.grad is not None else torch.zeros_like(x0)

#         if iteration == 0:
#             initial_guess = u_field.detach().clone().cpu()
#             plot_single(initial_guess.squeeze(),
#                         f'{folder}/iter={batch_num}_inversion_{iteration}_initial_guess.png',
#                         vmin=prior.min().item(), vmax=prior.max().item(), cmap='BrBG')
#             plot_single(prior.detach().cpu().squeeze(),
#                         f'{folder}/iter={batch_num}_inversion_{iteration}_prior.png',
#                         vmin=prior.min().item(), vmax=prior.max().item(), cmap='BrBG')

#         # diagnostics
#         inv_mse = torch.sqrt(torch.mean((u_field - prior.squeeze()) ** 2))
#         elapsed = time.time() - start_time

#         loss_data_iter.append({
#             "sample": batch_num,
#             "iteration": iteration,
#             "elapsed_s": elapsed,
#             "loss": loss_total.item(),
#             "inversion_MSE": inv_mse.item()
#         })

#         if iteration % 10 == 0:
#             print(f"[Iter {iteration:04d}] loss={loss_total.item():.3e}, ||grad||={g.norm().item():.3e}")

#         # Plot every save_fig_iter iterations
#         if iteration % save_fig_iter == 0:
#             u_np = u_field.detach().cpu().squeeze()
#             g_np = g.detach().cpu().reshape(ncmp, ncmp)
#             plot_single(u_np, f'{folder}/iter={batch_num}_inversion_{iteration}.png',
#                         vmin=prior.min().item(), vmax=prior.max().item(), cmap='BrBG')
#             plot_single(g_np, f'{folder}/iter={batch_num}_inversion_{iteration}_grad.png', cmap='PRGn')
#             plot_single(x0.detach().cpu().reshape(ncmp, ncmp),
#                         f'{folder}/iter={batch_num}_inversion_{iteration}_coeff.png',
#                         vmin=-3.0, vmax=4.0, cmap='plasma')

#         # Append history
#         posterior_set.append(u_field.detach().cpu().numpy())
#         output_set.append(y_pred.detach().cpu().numpy())
#         gradient_set.append(g.detach().cpu().reshape(ncmp, ncmp).numpy())

#     print(f"[Done] Sample {batch_num} finished after {num_iterations} iterations.")

#     return posterior_set, y_pred, loss_data_iter, i, j, curves, folder, output_set, gradient_set, input_data

# def least_squares_posterior_estimation_prev(
#     model, input_data, true_data, learning_rate,
#     batch_num, num_iterations=500, prior=None,
#     i=None, j=None, folder=None):
#     """
#     Gradient-descent inversion where we directly update the PHYSICAL FIELD (u_field)
#     instead of the Fourier coefficients.
#     Keeps same plotting, logging, and return values.
#     """

#     if opt_type != "Devito" or data_type != "Darcy":
#         model.eval()

#     device = input_data.device
#     true_data = true_data.to(device)
#     sigma_y = 1e-2
#     mse_loss = torch.nn.MSELoss(reduction='mean')

#     # ------------------------------------------------------------
#     # Initialize optimization variable: physical field (u_field)
#     # ------------------------------------------------------------
#     u_field = input_data.clone().detach().requires_grad_(True)
#     optimizer = torch.optim.SGD([u_field], lr=learning_rate, momentum=0.0)

#     # Containers for diagnostics
#     posterior_set, output_set, curves = [], [], []
#     losses, inversion_MSEs, regs, ssims, infty_norm, gradient_set, loss_data_iter = [], [], [], [], [], [], []
#     start_time = time.time()

#     print(f"[Start GD inversion on field] lr={learning_rate}, iterations={num_iterations}")

#     for iteration in range(num_iterations):
#         optimizer.zero_grad()

#         # ----------------------
#         # Forward PDE / surrogate
#         # ----------------------
#         if opt_type == "Devito":
#             y_pred = model(u_field.squeeze())
#         else:
#             # ensure shape (B, C, H, W)
#             y_pred = model(u_field.unsqueeze(0).unsqueeze(0))

#         # ----------------------
#         # Data misfit loss
#         # ----------------------
#         if full_obs:
#             pred = y_pred.flatten()
#             target = true_data.flatten()
#         else:
#             pred = y_pred.squeeze()[i, j]
#             target = true_data.squeeze()[i, j]

#         loss_total = ((pred - target) / sigma_y).pow(2).mean()
#         loss_total.backward()
#         optimizer.step()

#         # ----------------------
#         # Clamp field to valid range
#         # ----------------------
#         with torch.no_grad():
#             min_clamp = prior.min().item() - 0.01
#             max_clamp = prior.max().item() + 0.01
#             u_field.clamp_(min=min_clamp, max=max_clamp)

#         # ----------------------
#         # Record diagnostics
#         # ----------------------
#         g = u_field.grad.detach().clone() if u_field.grad is not None else torch.zeros_like(u_field)

#         if iteration == 0:
#             initial_guess = u_field.detach().clone().cpu()
#             plot_single(initial_guess.squeeze(),
#                         f'{folder}/iter={batch_num}_inversion_{iteration}_initial_guess.png',
#                         vmin=prior.min().item(), vmax=prior.max().item(), cmap='BrBG')
#             plot_single(prior.detach().cpu().squeeze(),
#                         f'{folder}/iter={batch_num}_inversion_{iteration}_prior.png',
#                         vmin=prior.min().item(), vmax=prior.max().item(), cmap='BrBG')

#         inv_mse = torch.sqrt(torch.mean((u_field - prior.squeeze()) ** 2))
#         elapsed = time.time() - start_time

#         loss_data_iter.append({
#             "sample": batch_num,
#             "iteration": iteration,
#             "elapsed_s": elapsed,
#             "loss": loss_total.item(),
#             "inversion_MSE": inv_mse.item()
#         })

#         if iteration % 10 == 0:
#             print(f"[Iter {iteration:04d}] loss={loss_total.item():.3e}, ||grad||={g.norm().item():.3e}")

#         # ----------------------
#         # Plot every few iterations
#         # ----------------------
#         if iteration % save_fig_iter == 0:
#             u_np = u_field.detach().cpu().squeeze()
#             g_np = g.detach().cpu().squeeze()
#             plot_single(u_np,
#                         f'{folder}/iter={batch_num}_inversion_{iteration}.png',
#                         vmin=prior.min().item(), vmax=prior.max().item(), cmap='BrBG')
#             plot_single(g_np,
#                         f'{folder}/iter={batch_num}_inversion_{iteration}_grad.png',
#                         cmap='PRGn')

#         posterior_set.append(u_field.detach().cpu().numpy())
#         output_set.append(y_pred.detach().cpu().numpy())
#         gradient_set.append(g.detach().cpu().squeeze().numpy())

#     print(f"[Done] Sample {batch_num} finished after {num_iterations} iterations.")

#     return posterior_set, y_pred, loss_data_iter, i, j, curves, folder, output_set, gradient_set, initial_guess


def least_squares_posterior_estimation_prev_(model, input_data, true_data, learning_rate,
                                       batch_num, num_iterations=500, prior=None, i=None, j=None, folder=None):
    if opt_type != "Devito" or data_type != "Darcy":
        model.eval()

    x0 = input_data.clone().detach().requires_grad_(True).to(device)
    posterior_set, output_set, curves = [], [], []
    losses, inversion_MSEs, regs, ssims, infty_norm, gradient_set, loss_data_iter = [], [], [], [], [], [], []

    start_time = time.time()
    true_data = true_data.to(device)
    if data_type == "NS":
        max_clamp = prior.max().item() + 0.1
        min_clamp = prior.min().item() - 0.1
        mse_loss = torch.nn.MSELoss(reduction='mean')
        if opt_type == "Devito":
            x0 = x0.double()
            true_data = true_data.double()
    else:
        max_clamp = prior.max().item() + 0.01 #0.91
        min_clamp = prior.min().item() - 0.01 #0.09
        mse_loss = torch.nn.MSELoss(reduction='sum') #sum

    if opt_type == "Devito" and data_type == "Darcy":
        if full_obs == True:
            extracted_target = true_data[:, :, i, j]
        else:
            extracted_target = true_data[:, :, i, j]
    else:
        if full_obs:
            extracted_target = true_data.flatten()
        else:
            extracted_target = true_data[:, :, i, j]
    plot_single(true_data[0].detach().cpu().squeeze(), "sanitycheck_y.png", "jet", vmin=true_data[0].min())

    for iteration in range(num_iterations):
        if iteration == 0:
            x0 = input_data.clone().detach().requires_grad_(True).to(device)
        else:
            x0 = x0.detach().clone().requires_grad_(True).to(device)

        x0.grad = None
        x0_old = x0.detach().clone()

        if opt_type == "Devito" and data_type == "Darcy":
            squeezed_x0 = x0.squeeze()
            squeezed_x0.retain_grad()
            output = model(squeezed_x0)
        else:
            print("x0shape", x0.shape)
            output = model(x0.reshape(1,1,dim,dim))

        # extract and compute loss
        if opt_type == "Devito" and data_type == "Darcy":
            extracted_output = output[i, j]
        else:
            extracted_output = output[:, :, i, j]
            print("extracted output", extracted_output.shape)

        def loss_fn(x):
            with torch.no_grad():
                # 1) forward
                # target = true_data[0, 0, i, j]
                if opt_type == "Devito" and data_type == "Darcy":
                    out = model(x.squeeze())
                    if full_obs:
                        pred = out.flatten() # extract observed entries
                    else:
                        pred = out[i,j]
                else:
                    # FNO / PyTorch path: [1,1,H,W] → [1,1,H,W]
                    out = model(x)
                    if full_obs:
                        pred = out.flatten()
                    else:
                        pred = out[0, 0, i, j]
                # 2) data misfit
                data_misfit = mse_loss(pred.squeeze(), extracted_target.squeeze())
                # 4) combine and return as Python float
                return float(data_misfit.item())



        loss = mse_loss(extracted_output.squeeze(), extracted_target.squeeze())
        reg = 0. #sobolev_H1_norm(x0)
        loss_total = loss #+ alpha * reg
        loss_total.backward()
        g = x0.grad.detach()

        if type_opt == "NGD":
            with torch.no_grad():
                if opt_type == "Devito" and data_type == "Darcy":
                    Q = fisher_approx_vjp_batched(groundwater_model, x0, i, j, noise_std,rank=400,chunk_size=80,opt_type=opt_type,forcing_term=forcing_term, full_obs=full_obs)
                else:
                    Q = fisher_approx_vjp_batched(model, x0, i, j, noise_std, rank=400, chunk_size=100, full_obs=full_obs)#250

            # Precondition gradient
            '''g = x0.grad.detach().flatten()
            B = Q.T @ Q  # [r × r]
            natural_grad = Q @ torch.linalg.solve(B, Q.T @ g)
            g = natural_grad.reshape_as(x0)'''

            # ---- choose damping (can be constant or adaptive) ----
            lam = 1e-4 #1e-1                 # e.g. 1e-3; try 1e-4 … 1e-1 or make it adaptive
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
            x0.data = torch.clamp(x0.data, min=min_clamp, max=max_clamp)
            x0.requires_grad_(True)
            x0_new = x0.detach().clone()

        # metrics (L2)
        diff = x0 - prior
        inv_mse = torch.sqrt(mse_loss(x0, prior))

        # metrics (H1)
        sobolev_num   = sobolev_H1_norm(diff)          # ‖x - x★‖_H1
        sobolev_denom = sobolev_H1_norm(prior)         # ‖x★‖_H1
        rel_sobolev   = (sobolev_num / sobolev_denom).sqrt()  # take √ because norm fn returns squared integral

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
            "inversion_MSE": inv_mse.item(),
            "regularization":reg,
            "SSIM":          ssim_value,
            "rel_H1":        rel_sobolev.item()
        })

        print(f"loss={loss.item():.3e}, ||grad||={g.norm().item():.3e}, max|grad|={g.abs().max().item():.3f}")

        # if batch_num < 2 and iteration % 50 == 0 and opt_type != "Devito":
        if batch_num < 2 and iteration % save_fig_iter == 0:
            gradient = g.detach().cpu().squeeze()  # shape: [H, W] or similar
            plt.imshow(gradient.numpy(), cmap='viridis')
            plt.colorbar(shrink=0.8)
            plt.tight_layout()
            plt.title('Gradient w.r.t. Input x0')
            plt.savefig(f'{folder}/iter={batch_num}_gradient_{iteration}.png')
            plot_single(x0.detach().cpu().squeeze(), f'{folder}/iter={batch_num}_inversion_{iteration}.png', vmin=min_clamp)
            plot_single(output.detach().cpu().squeeze(), f'{folder}/iter={batch_num}_inversion_{iteration}_output.png', "jet")
            if data_type == "Darcy": 
                plot_single(darcy_mask2(x0.detach().cpu().squeeze()), f'{folder}/thresholded_a.png', vmin=min_clamp)
            plot_inversion_result(zero_X, x, true_data.detach().cpu().squeeze(), output.detach().cpu().squeeze(), x0.clone().detach().cpu().numpy(), opt_type, batch_num, i, j, iteration, folder, data_type, sub_sampling)
        
        print(f"Iteration {iteration}, Loss: {loss_total.item():.4e}", inv_mse.item(), ssim_value)

        # store for plotting later
        posterior_set.append(x0.clone().detach().cpu().numpy())
        output_set.append(output.clone().detach().cpu().numpy())
        gradient_set.append(g.cpu().numpy())
        plot_single(g.cpu().numpy().reshape(dim,dim), "sanity_check_grad.png")


    return posterior_set, output, loss_data_iter, i, j, curves, folder, output_set, gradient_set, zero_X



def least_squares_posterior_estimation_wrt(model, input_data, true_data, learning_rate,
                                       batch_num, num_iterations=500, prior=None, i=None, j=None, folder=None):
    if opt_type != "Devito" or data_type != "Darcy":
        model.eval()

    x0 = input_data.clone().detach().requires_grad_(True).to(device)
    posterior_set, output_set, curves = [], [], []
    losses, inversion_MSEs, regs, ssims, infty_norm, gradient_set, loss_data_iter = [], [], [], [], [], [], []

    start_time = time.time()
    true_data = true_data.to(device)
    if data_type == "NS":
        max_clamp = prior.max().item() + 0.1
        min_clamp = prior.min().item() - 0.1
        mse_loss = torch.nn.MSELoss(reduction='mean')
        if opt_type == "Devito":
            x0 = x0.double()
            true_data = true_data.double()
    else:
        max_clamp = prior.max().item() + 0.01 #0.91
        min_clamp = prior.min().item() - 0.01 #0.09
        mse_loss = torch.nn.MSELoss(reduction='sum')

    if opt_type == "Devito" and data_type == "Darcy":
        if full_obs == True:
            extracted_target = true_data[:, :, i, j]
        else:
            extracted_target = true_data[:, :, i, j]
    else:
        if full_obs:
            extracted_target = true_data.flatten()
        else:
            extracted_target = true_data[:, :, i, j]
    plot_single(true_data[0].detach().cpu().squeeze(), "sanitycheck_y.png", "jet")

    for iteration in range(1, num_iterations, 100):
        # Load simulator iterate for this sample and iteration
        x0_np = dset_ns[batch_num, iteration, :, :]   # numpy (128,128)
        x0 = torch.tensor(x0_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        x0.requires_grad_(True)

        x0.grad = None
        x0_old = x0.detach().clone()

        if opt_type == "Devito" and data_type == "Darcy":

            # new ----------------------------------------
            squeezed_x0 = x0.squeeze()
            # squeezed_x0 = darcy_mask_tanh(x0.squeeze())
            squeezed_x0.retain_grad()
            output = model(squeezed_x0)
        # elif data_type == "Darcy" and opt_type != "Devito":
        #     squeezed_x0 = darcy_mask_tanh(x0.squeeze())
        #     squeezed_x0.retain_grad()
        #     output = model(squeezed_x0)
        else:
            output = model(x0)

        # extract and compute loss
        if opt_type == "Devito" and data_type == "Darcy":
            extracted_output = output[i, j]
        else:
            extracted_output = output[:, :, i, j]

        loss = mse_loss(extracted_output.squeeze(), extracted_target.squeeze())
        reg = 0. #sobolev_H1_norm(x0)
        loss_total = loss #+ alpha * reg
        loss_total.backward()
        g = x0.grad.detach()

        with torch.no_grad():
            x0 -= learning_rate * g
            x0.data = torch.clamp(x0.data, min=min_clamp, max=max_clamp)
            x0.requires_grad_(True)
            x0_new = x0.detach().clone()

        input_numpy = x0.detach().cpu().squeeze().numpy()
        prior_numpy = prior.detach().cpu().squeeze().numpy()
        # elapsed time
        now = time.time()
        elapsed = now - start_time

        # record
        loss_data_iter.append({
            "sample":        batch_num,
            "iteration":     iteration,
            "elapsed_s":     elapsed
        })

        print(f"loss={loss.item():.3e}, ||grad||={g.norm().item():.3e}, max|grad|={g.abs().max().item():.3f}")

        # if batch_num < 2 and iteration % 50 == 0 and opt_type != "Devito":
        if batch_num < 2 and iteration % save_fig_iter == 0:
            gradient = g.detach().cpu().squeeze()  # shape: [H, W] or similar
            plt.imshow(gradient.numpy(), cmap='viridis')
            plt.colorbar(shrink=0.8)
            plt.tight_layout()
            plt.title('Gradient w.r.t. Input x0')
            plt.savefig(f'{folder}/iter={batch_num}_gradient_{iteration}.png')

            plot_single(x0.detach().cpu().squeeze(), f'{folder}/iter={batch_num}_inversion_{iteration}.png', vmin=min_clamp)
            plot_single(output.detach().cpu().squeeze(), f'{folder}/iter={batch_num}_inversion_{iteration}_output.png', "jet")
            if data_type == "Darcy": 
                plot_single(darcy_mask2(x0.detach().cpu().squeeze()), f'{folder}/thresholded_a.png', vmin=min_clamp)
            plot_inversion_result(zero_X, x, y, output.detach().cpu().squeeze(), x0.clone().detach().cpu().numpy(), opt_type, batch_num, i, j, iteration, folder, data_type, sub_sampling)
        
        print(f"Iteration {iteration}, Loss: {loss_total.item():.4e}")

        # store for plotting later
        posterior_set.append(x0.clone().detach().cpu().numpy())
        output_set.append(output.clone().detach().cpu().numpy())
        gradient_set.append(g.cpu().numpy())
        plot_single(g.cpu().numpy().reshape(dim,dim), f"sanity_check_grad_wrt_{iteration}.png")


    return posterior_set, output, loss_data_iter, i, j, curves, folder, output_set, gradient_set



# ------------------------------------------------------------
# Main Script for Inversion on Multiple Samples (batch_size=1)
# ------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print(f"Using device: {device}")
    
    '''Experimental Factor'''
    num_vec = 64                   # [50, 200, 400]
    opt_type = "JAC"             # ["JAC", "MSE", "Devito", "PINO"]
    initial_guess = "prior_mean"    # ["smooth", "noisy", "prior_mean"]
    data_type = "Darcy"             # ["Darcy", "NS", "VMB"]

    '''Type of Optimizer'''
    type_opt = "GD"                 # ["GD", "NGD"]

    '''Experimental Factor on Observation'''
    noise_std = 0.01                # [0.01, 0.02, 0.05]
    sub_sampling = True             # [True, False]
    full_obs = not sub_sampling
    top_subsampling = False         
    ood_prior = False
    ood_tau = 18.00                 # [3.05, 5.0, 18.0]
    subsample_ratio = 0.25          # [0.25, 0.10, 0.05, 0.02]

    '''Other Experimental Factor'''
    wrt_NS = False   
    wrt_NS_type = f"Devito_NS" #["JAC_Darcy_400, MSE_Darcy", "MSE_NS", "JAC_NS_400"]
    fname_source = "noise(0.01)_NS_smooth_partial(0.1)_lr=0.5_sample=10"
    fname_ns = f"{fname_source}/inversion_history_{wrt_NS_type}_{initial_guess}_GD.h5"
    
    '''Ploting Factor'''
    spectral = False
    print("type opt", type_opt)

    if initial_guess == "prior_mean" and data_type == "Darcy":
        learning_rate = 5e-2 #0.2        # [0.01, 5, 30000, 0.5]
        num_sample = 1              # [10, 2, 1]
        num_sample_prior = 100       # [50, 100]
        num_epoch = 31            # [801, 2501, 4001]
        kernel_size = 49              # 31
        sigma = 80.0
        save_fig_iter = 5
        decay_interval = 2000
        dim = 128
        alpha = 0.
        subsample_ratio = 0.04
        noise_std = 0.01
    elif initial_guess == "smooth" and data_type == "Darcy":
        learning_rate = 0.5        # [0.01, 0.005, 0.001]
        num_sample = 1              # [5, 2, 1]
        num_sample_prior = 200 
        num_epoch = 2001            # [4001, 6001]
        kernel_size = 21            # [45, 35, 25, 19]
        sigma = 30.0                # [100.0, 50.0, 30.0]
        save_fig_iter = 50
        decay_interval = 20         # [20]
        dim = 128
        alpha = 0.
        subsample_ratio = 0.1
    elif initial_guess == "smooth" and data_type == "NS":
        learning_rate = 0.05 #1e-7 #2 # 0.01 #3000 #5000 @TODO
        num_sample = 10 #2 #1
        num_sample_prior = 100 #200, 40 # 30, 50 #5 #@TODO
        num_epoch = 2 #2501 #2501 #2001 @TODO
        sub_sampling = True
        full_obs = not sub_sampling
        subsample_ratio = 0.25
        save_fig_iter = 50
        decay_interval = 20
        dim = 64
        kernel_size = 35
        sigma = 50.0
        alpha = 0. #0.00001
    elif data_type == "VMB":
        sub_sampling == False

    
    # ----------------------------------------------
    # Load configuration and dataset. and checkpoint
    # ----------------------------------------------
    if data_type == "Darcy":
        if opt_type == "JAC" and num_vec == 64:
            config = load_config("configs/eigenvectors/e_50.yaml")
            ckpt_path = "checkpoints/n=400_e=64_m=FNO_s=RFS_l=JAC_20260102_152517/n=400_e=64_m=FNO_s=RFS_l=JAC_epoch=199_val_rel_l2_loss=0.0095.ckpt"
        elif opt_type == "JAC" and num_vec == 200:
            config = load_config("configs/eigenvectors/e_200.yaml")
            # ckpt_path = f"checkpoints/n=400_e=200_m=FNO_s=RFS_l=JAC_20250615_133916/n=400_e=200_m=FNO_s=RFS_l=JAC_epoch=190_val_rel_l2_loss=0.0170.ckpt"
            # ckpt_path = "checkpoints/n=400_e=200_m=FNO_s=RFS_l=JAC_20251027_154534/n=400_e=200_m=FNO_s=RFS_l=JAC_epoch=133_val_rel_l2_loss=0.0154.ckpt"
            # ckpt_path = "checkpoints/n=400_e=200_m=FNO_s=RFS_l=JAC_20251101_214222/n=400_e=200_m=FNO_s=RFS_l=JAC_epoch=121_val_rel_l2_loss=0.0108.ckpt"
            ckpt_path = "checkpoints/n=400_e=200_m=FNO_s=RFS_l=JAC_20251101_214222/n=400_e=200_m=FNO_s=RFS_l=JAC_epoch=198_val_rel_l2_loss=0.0095.ckpt"
        elif opt_type == "JAC" and num_vec == 400:
            config = load_config("configs/eigenvectors/e_400.yaml")
            # ckpt_path = f"checkpoints/n=400_e=400_m=FNO_s=RFS_l=JAC_20250617_131205/n=400_e=400_m=FNO_s=RFS_l=JAC_epoch=204_val_rel_l2_loss=0.0172.ckpt"
        elif opt_type == "MSE":
            config = load_config("configs/darcy_MSE.yaml")
            # ckpt_path = "checkpoints/DARCY_MSE/Darcy_training_epoch=249_val_rel_l2_loss=0.0009_MSE_May14.ckpt"
            # ckpt_path = "checkpoints/Darcy_training_20250615_133632/Darcy_training_epoch=123_val_rel_l2_loss=0.0173.ckpt"
            # ckpt_path = "checkpoints/Darcy_training_20251026_185639/Darcy_training_epoch=105_val_rel_l2_loss=0.0156.ckpt"
            # ckpt_path = "checkpoints/Darcy_training_20251101_214307/Darcy_training_epoch=092_val_rel_l2_loss=0.0110.ckpt"
            ckpt_path = "checkpoints/Darcy_training_20251101_214307/Darcy_training_epoch=172_val_rel_l2_loss=0.0096.ckpt"
        elif opt_type == "PINO":
            config = load_config("configs/darcy_MSE.yaml")
            ckpt_path = "../../model/PINO_darcy_best_2.7040e-02_1st_try.pt"

        # Numerical Simulator
        forcing_term = torch.ones(128,128) #torch.zeros(128, 128)
        gw_torch_model = GroundwaterModel(forcing_term.shape[0]) 
        def forward_with_tsteps(u, f, time_steps=50000):
            eq = gw_torch_model.groundwater_eq
            orig = eq.eval_fwd_op

            def wrapped(f_, u_, *args, **kwargs):
                kwargs.pop("time_steps", None)
                return orig(f_, u_, time_steps=time_steps, *args, **kwargs)

            eq.eval_fwd_op = wrapped
            try:
                return gw_torch_model(u, f)
            finally:
                eq.eval_fwd_op = orig



    elif data_type == "NS":
        if opt_type == "JAC" and num_vec == 200:
            config = load_config("configs/eigenvectors/e_400_NS_new.yaml")
            ckpt_path = f"checkpoints/n=1000_e=200_m=FNO_s=RFS_l=JAC_20250903_013609/n=1000_e=200_m=FNO_s=RFS_l=JAC_epoch=200_val_rel_l2_loss=0.1504.ckpt"
        elif opt_type == "JAC" and num_vec == 400:
            config = load_config("configs/eigenvectors/e_400_NS_new.yaml")
            # ckpt_path = f"checkpoints/n=400_e=400_m=FNO_s=RFS_l=JAC_20250907_142507/n=400_e=400_m=FNO_s=RFS_l=JAC_epoch=183_val_rel_l2_loss=0.1652.ckpt"
            # ckpt_path = "checkpoints/n=400_e=400_m=FNO_s=RFS_l=JAC_20250907_142507/n=400_e=400_m=FNO_s=RFS_l=JAC_epoch=170_val_rel_l2_loss=0.1650.ckpt"
            ckpt_path = "checkpoints/n=400_e=400_m=FNO_s=RFS_l=JAC_20251015_130613/n=400_e=400_m=FNO_s=RFS_l=JAC_epoch=210_val_rel_l2_loss=0.0591.ckpt"
        elif opt_type == "MSE":
            config = load_config("configs/eigenvectors/e_0_NS_new.yaml")
            ckpt_path = "checkpoints/n=400_m=FNO_l=L2_20251015_125349/n=400_m=FNO_l=L2_epoch=381_val_rel_l2_loss=0.0594.ckpt"
            # ckpt_path = "checkpoints/n=1000_m=FNO_l=L2_20250915_114551/n=1000_m=FNO_l=L2_epoch=122_val_rel_l2_loss=0.1655.ckpt"

    # Surrogate model OR numerical simulator
    if opt_type != "Devito" and opt_type != "PINO":
        model = NSModel.load_from_checkpoint(ckpt_path).eval().to(device)
        with open("rng_state_devito.pkl", "rb") as f:
            state = pickle.load(f)
            np.random.set_state(state["np_random_state"])
            random.setstate(state["random_state"])
    elif opt_type == "PINO":
        model = get_model(config.experiment.model_type, config.model_settings).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    elif opt_type == "Devito" and data_type == "Darcy":
        # model = lambda x: gw_torch_model(x, forcing_term)
        model = lambda x: forward_with_tsteps(x, forcing_term, time_steps=50000)
        groundwater_model = GroundwaterEquation(forcing_term.shape[0])
    elif opt_type == "Devito" and data_type == "NS":
        model = NavierStokesSimulator(dim, dim, 100, 2.0, 1e-3)

    # Load Data
    if data_type == "Darcy":
        # data_config = load_config("output/n=400_e=200_m=FNO_s=RFS_l=JAC_20250615_133916/config.yaml")
        data_config = load_config("configs/eigenvectors/e_200.yaml")
        dim_g = 8
    elif data_type == "NS":
        data_config = load_config("configs/eigenvectors/e_400_NS_new.yaml")
        dim_g = 64

    data_config.data_settings['batch_size'] = 1
    dataset = get_dataset(data_config.experiment.dataset_type, data_config.data_settings)
    dataloader = dataset.get_dataloader(offset=405, limit=num_sample)
    data_config.data_settings['batch_size'] = 50
    dataset = get_dataset(data_config.experiment.dataset_type, data_config.data_settings)
    prior_dataloader = dataset.get_dataloader(offset=414, limit=num_sample_prior)

    # Save RNG state
    with open("rng_state_devito.pkl", "wb") as f:
        pickle.dump({
            "np_random_state": np.random.get_state(),
            "random_state": random.getstate()
        }, f)

    # Initialize a list to hold loss and metric data for each sample.
    loss_data_all, metrics_all, final_ssim_list, final_l2_list = [], [], [], []
    sample_counter = 0

    # Construct folder
    if opt_type != 'JAC':
        if wrt_NS:
            folder_name = f'{fname_source}/inversion_result_{opt_type}_{data_type}_{initial_guess}_{type_opt}_{wrt_NS_type}'
            fname = f'{fname_source}/inversion_history_{opt_type}_{data_type}_{initial_guess}_{type_opt}_{wrt_NS_type}.h5'
            fname_output = f'{fname_source}/inversion_history_output_{opt_type}_{data_type}_{initial_guess}_{type_opt}_{wrt_NS_type}.h5'
        else:
            folder_name = f'inversion_result_{opt_type}_{data_type}_{initial_guess}_{type_opt}'
            fname = f'inversion_history_{opt_type}_{data_type}_{initial_guess}_{type_opt}.h5'
            fname_output = f'inversion_history_output_{opt_type}_{data_type}_{initial_guess}_{type_opt}.h5'
    else:
        if wrt_NS:
            folder_name = f'{fname_source}/inversion_result_{opt_type}_{data_type}_{num_vec}_{initial_guess}_{type_opt}_{wrt_NS_type}'
            fname = f'inversion_history_{opt_type}_{data_type}_{num_vec}_{initial_guess}_{type_opt}_{wrt_NS_type}.h5'
            fname_output = f'inversion_history_output_{opt_type}_{data_type}_{num_vec}_{initial_guess}_{type_opt}_{wrt_NS_type}.h5'
        else:
            folder_name = f'inversion_result_{opt_type}_{data_type}_{num_vec}_{initial_guess}_{type_opt}'
            fname = f'inversion_history_{opt_type}_{data_type}_{num_vec}_{initial_guess}_{type_opt}.h5'
            fname_output = f'inversion_history_output_{opt_type}_{data_type}_{num_vec}_{initial_guess}_{type_opt}.h5'
    os.makedirs(folder_name, exist_ok=True)

    # construct files for saving gradient
    if wrt_NS:
        if opt_type == "JAC":
            fname_gradient = f'{fname_source}/inversion_history_gradient_NS_{opt_type}_{data_type}_{num_vec}_{initial_guess}_{type_opt}.h5'
        elif opt_type != "JAC":
            fname_gradient = f'{fname_source}/inversion_history_gradient_NS_{opt_type}_{data_type}_{initial_guess}_{type_opt}_{wrt_NS_type}.h5'
        else:
            fname_gradient = f'{fname_source}/inversion_history_gradient_NS_{opt_type}_{data_type}_{initial_guess}_{type_opt}_{wrt_NS_type}.h5'
    else:
        if opt_type == "JAC":
            fname_gradient = f'inversion_history_gradient_{opt_type}_{data_type}_{num_vec}_{initial_guess}_{type_opt}.h5'
        elif opt_type != "JAC":
            fname_gradient = f'inversion_history_gradient_{opt_type}_{data_type}_{initial_guess}_{type_opt}.h5'
        else:
            fname_gradient = f'inversion_history_gradient_{opt_type}_{data_type}_{initial_guess}_{type_opt}_top.h5'

    # If it already exists, delete it (and any stale lock)
    if os.path.exists(fname): os.remove(fname)
    if os.path.exists(fname_output): os.remove(fname_output)
    if os.path.exists(fname_gradient): os.remove(fname_gradient)
    # Now create it
    h5_file = h5py.File(fname, 'w')
    h5_file_output = h5py.File(fname_output, 'w')
    h5_file_gradient = h5py.File(fname_gradient, 'w')
    num_samples = len(dataloader)
    if wrt_NS:
        length = int((num_epoch-1) / 100)
    else:
        length = num_epoch
    dset = h5_file.create_dataset(
        'a', shape=(num_samples, length, dim, dim),
        dtype='f4', compression='gzip',
        compression_opts=4, chunks=(1, length, dim, dim)  # chunk by sample
    )
    dset_output = h5_file_output.create_dataset(
        'u', shape=(num_samples, length, dim, dim),
        dtype='f4', compression='gzip',
        compression_opts=4, chunks=(1, length, dim, dim)
    )
    dset_gradient = h5_file_gradient.create_dataset(
        'g', shape=(num_samples, length, dim_g, dim_g),
        dtype='f4', compression='gzip',
        compression_opts=4, chunks=(1, length, dim_g, dim_g)
    )

    # Compute prior mean
    if initial_guess == "prior_mean":
        total_x = 0.0
        num_samples = 0

        for batch in prior_dataloader:
            x = batch['x'].to(device)
            x_101 = x[33] #33
            zero_X = x_101.unsqueeze(0)
            x[x >= 0.5] = 0.9
            x[x < 0.5] = 0.1
            total_x += x.sum(dim=0)       # accumulate sum over batch dimension
            print("total_x", total_x.shape)
            num_samples += x.shape[0]     # count number of samples

        # Average over all samples
        prior_mean = total_x / num_samples
        prior_mean = prior_mean.unsqueeze(0).detach()   # add batch dim
        print("shape of prior mean:", prior_mean.shape)

    if wrt_NS:
        print("wrt_NS file name", fname_ns)
        h5_file_ns = h5py.File(fname_ns, 'r')
        dset_ns = h5_file_ns['a']                       # shape [num_samples, num_epoch, 128, 128]

    for batch in dataloader:

        if data_type != "Darcy":
            x = batch['x'].to(device)
            y = batch['y'].to(device)
        else:
            fname_darcy = "inv_Darcy/ellipticPDE_Beskos_0.h5"   # path to your saved file
            with h5py.File(fname_darcy, "r") as f:
                theta_truth = f["theta_truth"][:]
                u_sample = f["u_truth"][:]
                u_sample_exp = f["k_truth"][:]
                y = f["y_field"][:]
                cols = f["obs_pts"][:]
                cols = (cols * (128 - 1))
                cols = torch.tensor(cols).to(device)

            x = torch.tensor(u_sample).to(device).float().unsqueeze(0)
            y = torch.tensor(y).reshape(1, 1, dim, dim).float().to(device)
        
        plot_single(x.detach().cpu().squeeze(), f"sample_x{sample_counter}.png", "BrBG", vmin=x.min())
        plot_single(y.detach().cpu().squeeze(), f"sample_y{sample_counter}.png", "RdYlBu", vmin=y.min())
        V = batch['v'].to(device)
        Jvp = batch['Jvp'].to(device)
        L = batch['L'].view(-1).to(device)
        d = int(x.shape[-1])
        cols = torch.tensor([ (idx.item() // d, idx.item() % d) for idx in L ], device=device)
        i = cols[:, 0].long()
        j = cols[:, 1].long()

        if sub_sampling == True:
            if initial_guess == "smooth" or initial_guess == "prior_mean":
            #     if opt_type == "Devito":
            #         num_total = dim * dim
            #         indices = torch.randperm(num_total)[:int(subsample_ratio * num_total)]
            #         torch.save(indices, "subsample_indices.pt")
            #     else:
            #         indices = torch.load("subsample_indices.pt")
                
            #     coords = torch.stack(torch.meshgrid(
            #         torch.arange(dim),torch.arange(dim),
            #         indexing="ij"), dim=-1).reshape(-1, 2)
            #     selected_coords = coords[indices]
            #     # Build mask
            #     final_mask = torch.zeros((dim, dim), dtype=torch.bool)
            #     final_mask[selected_coords[:, 0], selected_coords[:, 1]] = True
            #     # Extract i, j like before
            #     i, j = final_mask.nonzero(as_tuple=True)
            #     print(i.shape, j.shape)
            #     print("selected", selected_coords)
            #     print("original", i)
            #     print("j", j)
                i, j = i, j
            else:
                i = i
                j = j
        elif full_obs == True:
            print("in full obs")
            i, j = torch.meshgrid(
                torch.arange(dim, device=device),
                torch.arange(dim, device=device),
                indexing='ij'
            )
            i = i.reshape(-1)
            j = j.reshape(-1)

        if initial_guess == "smooth":
            zero_X = apply_gaussian_smoothing(x.detach().clone(), kernel_size, sigma)
        elif initial_guess == "prior_mean":
            # zero_X = torch.zeros_like(prior_mean).detach()
            print("shape", theta_truth.shape)
            zero_X = apply_gaussian_smoothing(torch.tensor(u_sample).to(device).reshape(1, 1, 128,128), kernel_size, sigma)
            print("zero_X", zero_X.shape)
            # x_rand = 0.1 * torch.randn_like(x.detach().clone()).to(device)
            # zero_X, k_truth = reconstruct_fourier_field(x_rand, 128, 128, 4, 1.1, 1.0, 1.0, device)
            
            # zero_X = zero_X

        if opt_type == "Devito":
            # save data
            with h5py.File(f"{fname_source}/grf_sample_data_{data_type}_{initial_guess}_{sample_counter}.h5", "w") as f: #@TODO
                f.create_dataset("x", data=x.detach().cpu().numpy())
                f.create_dataset("y", data=y.detach().cpu().numpy())
                f.create_dataset("L", data=L.detach().cpu().numpy())
                f.create_dataset("i", data=i.detach().cpu().numpy())  # just the row indices
                f.create_dataset("j", data=j.detach().cpu().numpy())  # just the col indices
                f.create_dataset("zero_X", data=zero_X.detach().cpu().numpy())
                f.create_dataset("Jvp", data=Jvp.detach().cpu().numpy())
        else:
            # load data
            with h5py.File(f"{fname_source}/grf_sample_data_{data_type}_{initial_guess}_{sample_counter}.h5", "r") as f:
                x = torch.tensor(f["x"][:]).to(device)
                y = torch.tensor(f["y"][:]).to(device)
                L = torch.tensor(f["L"][:]).to(device)
                zero_X = torch.tensor(f["zero_X"][:]).to(device)
                i = torch.tensor(f["i"][:]).to(device).long()
                j = torch.tensor(f["j"][:]).to(device).long()

        # observation logic ...
        if ood_prior == True:
            # 1. define ood x
            if opt_type == "Devito":
                grf = GaussianRandomField(2, dim, alpha=2, tau=ood_tau)
                u_samples = grf.sample(1)
                # Sample random fields
                u_samples[u_samples>=0] = 0.9
                u_samples[u_samples<0] = 0.1
                x_prev = torch.tensor(u_samples[0])
                zero_X = (zero_X * num_sample_prior + x_prev.reshape(1, 1, dim, dim).cuda()) / (num_sample_prior + 1)
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
            y = y.reshape(1, 1, dim, dim).float()
            plot_single(x_prev.detach().cpu().squeeze(), f"ood_x.png", "viridis")
            plot_single(y.detach().cpu().squeeze(), f"ood_y.png", "jet")
        else:
            plot_single(y[0].detach().cpu().squeeze(), "sanitycheck_y0.png", vmin = y[0].min())
            y = y + torch.randn_like(y) * noise_std
            plot_single(y[0].detach().cpu().squeeze(), "sanitycheck_y2.png", vmin = y[0].min())

        plot_single(zero_X.detach().cpu().squeeze(), f"zero_X_sample_{sample_counter}.png", "viridis")

        if wrt_NS:
            posterior_set, pred, loss_data_iter, i_idx, j_idx, curves, folder, output_set, gradient_set = (
                least_squares_posterior_estimation_wrt(
                    model, zero_X, y, learning_rate, batch_num=sample_counter,
                    num_iterations=num_epoch, prior=x, i=i, j=j, folder=folder_name
                )
            )
        else:
            posterior_set, pred, loss_data_iter, i_idx, j_idx, curves, folder, output_set, gradient_set, zero_X = (
                least_squares_posterior_estimation_prev(
                    model, zero_X, y, learning_rate, batch_num=sample_counter,
                    num_iterations=num_epoch, prior=x, i=i, j=j, folder=folder_name
                )
            )

        # Plot the final inversion result.
        final_x0 = torch.tensor(posterior_set[-1]).cpu().numpy()
        plot_inversion_result(zero_X, x, y, pred.detach(), final_x0, opt_type, sample_counter, i_idx, j_idx, num_epoch, folder, data_type, sub_sampling)

        if spectral == True:
            if opt_type == "JAC":
                run_name = f"{opt_type}_{num_vec}_{initial_guess}_{type_opt}"   # e.g. "MSE" , "JVP50"
            else:
                run_name = f"{opt_type}_{initial_guess}_{type_opt}"   # e.g. "MSE" , "JVP50"
            k_all  = np.stack([c[0] for c in curves])   # shape (n_snapshots, n_bins)
            psd_all= np.stack([c[1] for c in curves])
            np.savez(f"psd_{run_name}.npz", k=k_all, psd=psd_all)

        # posterior_set is a list of length num_epoch, each an 128×128 numpy array. Write them into the HDF5 at [sample_counter, :, :, :]:
        arr = np.stack(posterior_set, axis=0).squeeze()   # shape (num_epoch,128,128)
        arr_output = np.stack(output_set, axis=0).squeeze()   # shape (num_epoch,128,128)
        arr_gradient = np.stack(gradient_set, axis=0).squeeze()
        dset[sample_counter, :, :, :] = arr
        dset_output[sample_counter, :, :, :] = arr_output
        dset_gradient[sample_counter, :, :, :] = arr_gradient # (101, 64)

        # collect this sample’s iteration‐by‐iteration records
        loss_data_all.extend(loss_data_iter)
        sample_counter += 1

    # save to single CSV
    df = pd.DataFrame(loss_data_all)
    # Close the HDF5 file
    h5_file.close()
    h5_file_output.close()
    h5_file_gradient.close()
    with h5py.File(fname, 'r') as f:
        print("On‑disk dataset shape is", f['a'].shape)

    if opt_type == "JAC" and wrt_NS == False:
        csv_file = f"loss_statistics_multiple_samples_{opt_type}_{data_type}_{num_vec}_{initial_guess}_{type_opt}.csv"
    elif opt_type != "JAC" and wrt_NS == False:
        csv_file = f"loss_statistics_multiple_samples_{opt_type}_{data_type}_{initial_guess}_{type_opt}.csv"
    else:
        csv_file = f"loss_statistics_multiple_samples_{opt_type}_{data_type}_{initial_guess}_{type_opt}_{wrt_NS_type}.csv"

    df.to_csv(csv_file, index=False)
    print(f"Loss data saved to {csv_file}")