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
from torch.autograd import grad
import pandas as pd

from models.ns_inversion import NSModel  # Your model
from utils import get_dataset, load_config, get_model  # Your utils

NUM_PSEUDO_TIMESTEPS: int = 500000



# ----------------------
# Plotting Functions
# ----------------------
def plot_single(true1, path, cmap="jet", vmin=None, vmax=None):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})
    print("vmin", vmin, vmax)
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


def compute_Jvp(model, x, v):
    Jvp = torch.zeros_like(v)
    for eig_idx in range(v.shape[-1]):
        v_i = v[:, :, :, eig_idx].unsqueeze(0)
        x = x.detach().clone().requires_grad_(True)
        # Use create_graph=False unless higher-order grads needed
        jvp_output, jvp_value = torch.autograd.functional.jvp(model, x, v_i, create_graph=False)
        Jvp[:, :, :, eig_idx] = jvp_value.detach()
        torch.cuda.empty_cache()
    x.requires_grad_(False)
    return jvp_output.detach(), Jvp.detach()


def compute_forward_and_gradient_errors(model, dataloader, device):
    model.eval()
    forward_errors = []
    gradient_errors = []

    for batch in dataloader:
        x = batch['x'].to(device).requires_grad_(True)  # shape: (1, 1, H, W)
        true_y = batch['y'].to(device)
        v = batch['v'].to(device)
        pred_y = model(x)

        # Compute forward L2 error
        forward_misfit = pred_y - true_y
        forward_error = torch.norm(forward_misfit.squeeze(), p=2) / torch.norm(true_y.squeeze(), p=2)
        forward_errors.append(forward_error.detach().cpu().item())

        # Compute gradient w.r.t. input
        # grad_pred = grad(pred_y.sum(), x, create_graph=False)[0]
        y_pred, grad_pred = compute_Jvp(model, x, v)
        grad_true = batch['Jvp'].to(device)
        misfit = grad_pred - grad_true
        grad_error = torch.norm(misfit.flatten(), p=2) / torch.norm(grad_true.flatten(), p=2)
        gradient_errors.append(grad_error.detach().cpu().item())

    avg_forward_error = np.mean(forward_errors)
    avg_gradient_error = np.mean(gradient_errors)

    print(f"Average Forward Error (L2): {avg_forward_error:.4e}")
    print(f"Average Gradient Error (L2): {avg_gradient_error:.4e}")

    return avg_forward_error, avg_gradient_error


def estimate_jacobian_rank(model, x, num_probes=100, tol=1e-6):
    """
    Estimate the rank of the Jacobian J = d(model(x)) / dx using matrix-free JvPs.
    Args:
        model: callable f(x)
        x: input tensor (requires_grad=True)
        num_probes: number of random vectors to probe
        tol: numerical threshold for singular value to be considered non-zero
    Returns:
        Estimated rank of J
    """
    # m = model(x, f).numel()
    Jv_matrix = []
    f = np.zeros_like(x)

    for _ in range(num_probes):
        v = torch.randn_like(x)
        if loss_type == "Devito":
            jvp = model.compute_linearization(f, x.cpu().numpy(), v.cpu().numpy())
            Jv_matrix.append(torch.tensor(jvp).flatten())
        else:
            y, jvp = torch.autograd.functional.jvp(model, x, v, create_graph=False)
            Jv_matrix.append(jvp.flatten())

    # Stack Jv outputs into a matrix: [m x num_probes]
    Jv_matrix = torch.stack(Jv_matrix, dim=1)  # shape: [m, num_probes]

    # Compute SVD of Jv_matrix (matrix-free sketch of J)
    U, S, V = torch.linalg.svd(Jv_matrix, full_matrices=False)

    # Count singular values above threshold
    rank = (S > tol).sum().item()
    return rank, S

# def estimate_jacobian_rank(model, x, num_probes=100, tol=1e-6):
#     """
#     Estimate the rank of the Jacobian J using randomized SVD via matrix-free JvPs.

#     Args:
#         model: callable f(x)
#         x: input tensor (requires_grad=True)
#         num_probes: rank parameter for randomized SVD
#         tol: threshold for singular value magnitude to count toward rank
#         loss_type: type of model (e.g. "Devito" for matrix-free linearization)

#     Returns:
#         Estimated rank and singular values
#     """
#     f = np.zeros_like(x)
#     Jv_matrix = []

#     # Step 1: Random projection matrix (implicitly via JvP)
#     for _ in range(num_probes):
#         v = torch.randn_like(x)
#         if loss_type == "Devito":
#             jvp = model.compute_linearization(f, x.cpu().numpy(), v.cpu().numpy())
#             Jv_matrix.append(torch.tensor(jvp).flatten())
#         else:
#             _, jvp = torch.autograd.functional.jvp(model, x, v, create_graph=False)
#             Jv_matrix.append(jvp.flatten())

#     # Step 2: Form sketch matrix Y = [J v_1, ..., J v_r]
#     Y = torch.stack(Jv_matrix, dim=1)  # Shape: [m, r]

#     # Step 3: QR decomposition Y = QR
#     Q, _ = torch.linalg.qr(Y, mode='reduced')  # Q: [m, r]

#     # Step 4: Project J onto low-rank subspace: B = Q^T J ≈ Q^T Y
#     B = Q.T @ Y  # Shape: [r, r]

#     # Step 5: Compute SVD of small matrix B
#     _, S, _ = torch.linalg.svd(B)

#     # Step 6: Estimate rank
#     rank = (S > tol).sum().item()

#     return rank, S



# ----------------------
# Main Script for Inversion on Multiple Samples (batch_size=1)
# ----------------------
if __name__ == "__main__":
    # Set up device and random seed.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print(f"Using device: {device}")


    # Define simulation parameters.
    num_vec = 1
    loss_type = "Devito"  # or "JAC" "MSE" "Devito"
    GRF = 2
    alpha = 0. #0.05
    noise_std = 0.1 #0.3
    initial_guess = "smooth" # "smooth", "noisy"
    sub_sampling = True
    top_subsampling = False
    full_obs = False
    num_sample = 100
    offset=120
    
    
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
        config = load_config("configs/eigenvectors/e_100.yaml")
        ckpt_path = f"checkpoints/DARCY_JAC_100/Darcy_training_epoch=249_val_rel_l2_loss=0.0022_JAC_May14.ckpt"
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
    if loss_type == "Devito":
        forcing_term = torch.zeros(128,128)
        model = GroundwaterEquation(forcing_term.shape[0])
        # groundwater_model = GroundwaterModel(forcing_term.shape[0])
        # model = lambda x: groundwater_model(x, forcing_term)

    # Load Data
    data_config = load_config("configs/eigenvectors/e_100.yaml")
    dataset = get_dataset(data_config.experiment.dataset_type, data_config.data_settings)
    dataloader = dataset.get_dataloader(offset=offset, limit=num_sample)

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

    print("Running forward experiment to evaluate surrogate model...")
    # compute_forward_and_gradient_errors(model, dataloader, device)
    # Get a single sample from the dataloader to use as input `x`
    batch = next(iter(dataloader))
    x = batch['x'].detach()

    rank, S = estimate_jacobian_rank(model, x, num_probes=3005, tol=1e-10)  # Set tol low to keep full spectrum

    # Choose your target ranks
    target_ranks = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000]

    # Sort S in decreasing order (largest singular values first)
    S_sorted = torch.sort(S, descending=True).values

    # Print tolerance needed for each target rank
    print(f"{'Target Rank':>12} | {'Min Tolerance':>15}")
    print("-" * 30)
    for r in target_ranks:
        if r <= len(S_sorted):
            tol_r = S_sorted[r - 1].item()
            print(f"{r:12d} | {tol_r:15.4e}")
        else:
            print(f"{r:12d} | {'(exceeds dim)':>15}")


#  Target Rank |   Min Tolerance
# ------------------------------
#           10 |      1.8262e+00
#           50 |      7.9751e-01
#          100 |      6.1228e-01
#          200 |      4.7409e-01
#          300 |      4.0817e-01
#          400 |      3.6472e-01
#          500 |      3.3085e-01
#          600 |      3.0284e-01
#          700 |      2.7809e-01
#          800 |      2.5479e-01
        #  900 |      2.3140e-01
        # 1000 |      1.9876e-01

#  Target Rank |   Min Tolerance
# ------------------------------
#           10 |      3.0062e+00
#           50 |      1.2967e+00
#          100 |      9.4475e-01
#          200 |      6.9979e-01
#          300 |      5.9411e-01
#          400 |      5.2907e-01
#          500 |      4.8254e-01
#          600 |      4.4739e-01
#          700 |      4.1846e-01
#          800 |      3.9435e-01
#          900 |      3.7277e-01
#         1000 |      3.5436e-01
#         2000 |      2.2666e-01
#         3000 |      1.2640e-01