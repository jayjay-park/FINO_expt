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
from utils_plot import plot_single

NUM_PSEUDO_TIMESTEPS: int = 500000



# ----------------------
# Plotting Functions
# ----------------------


def compute_Jvp(model, x, v):
    Jvp = torch.zeros_like(v)
    for eig_idx in range(v.shape[-1]):
        v_i = v[:, :, :, eig_idx].unsqueeze(1)
        x = x.detach().clone().requires_grad_(True)
        # Use create_graph=False unless higher-order grads needed
        jvp_output, jvp_value = torch.autograd.functional.jvp(model, x, v_i, create_graph=False)
        Jvp[:, :, :, eig_idx] = jvp_value.detach().squeeze()
        torch.cuda.empty_cache()
    x.requires_grad_(False)
    return jvp_output.detach(), Jvp.detach()


def compute_forward_and_gradient_errors(model, dataloader, device):
    model.eval()
    forward_errors = []
    gradient_errors = []
    mse_loss = torch.nn.MSELoss()

    for batch in dataloader:
        x = batch['x'].to(device).requires_grad_(True)  # shape: (1, 1, H, W)
        true_y = batch['y'].to(device)
        v = batch['v'].to(device)
        pred_y = model(x)

        # Compute forward L2 error
        forward_misfit = pred_y - true_y
        # forward_error = torch.norm(forward_misfit.squeeze().flatten(), p=2) / torch.norm(true_y.squeeze().flatten(), p=2)
        forward_error = mse_loss(pred_y.squeeze().flatten(), true_y.squeeze().flatten())
        forward_errors.append(forward_error.detach().cpu().item())

        # Compute gradient w.r.t. input
        # grad_pred = grad(pred_y.sum(), x, create_graph=False)[0]
        print("x", x.shape, "v", v.shape)
        y_pred, grad_pred = compute_Jvp(model, x, v)
        grad_true = batch['Jvp'].to(device)
        print("gradpred", grad_pred.shape, grad_true.shape)
        plot_single(grad_pred[0, :, :, 10].detach().cpu(), f"grad_pred_{loss_type}")
        plot_single(grad_true[0, :, :, 10].detach().cpu(), "grad_true")
        misfit = grad_pred - grad_true
        grad_error = mse_loss(grad_pred.flatten().squeeze(), grad_true.flatten().squeeze())
        # grad_error = torch.norm(misfit.flatten(), p=2) / torch.norm(grad_true.flatten(), p=2)
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

def estimate_jacobian_rank(model, x, num_probes=100, tol=1e-6):
    """
    Estimate the rank of the Jacobian J using randomized SVD via matrix-free JvPs.

    Args:
        model: callable f(x)
        x: input tensor (requires_grad=True)
        num_probes: rank parameter for randomized SVD
        tol: threshold for singular value magnitude to count toward rank
        loss_type: type of model (e.g. "Devito" for matrix-free linearization)

    Returns:
        Estimated rank and singular values
    """
    f = np.zeros_like(x)
    Jv_matrix = []

    # Step 1: Random projection matrix (implicitly via JvP)
    for _ in range(num_probes):
        v = torch.randn_like(x)
        if loss_type == "Devito":
            jvp = model.compute_linearization(f, x.cpu().numpy(), v.cpu().numpy())
            Jv_matrix.append(torch.tensor(jvp).flatten())
        else:
            _, jvp = torch.autograd.functional.jvp(model, x, v, create_graph=False)
            Jv_matrix.append(jvp.flatten())

    # Step 2: Form sketch matrix Y = [J v_1, ..., J v_r]
    Y = torch.stack(Jv_matrix, dim=1)  # Shape: [m, r]

    # Step 3: QR decomposition Y = QR
    Q, _ = torch.linalg.qr(Y, mode='reduced')  # Q: [m, r]

    # Step 4: Project J onto low-rank subspace: B = Q^T J ≈ Q^T Y
    B = Q.T @ Y  # Shape: [r, r]

    # Step 5: Compute SVD of small matrix B
    _, S, _ = torch.linalg.svd(B)

    # Step 6: Estimate rank
    rank = (S > tol).sum().item()

    return rank, S



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
    loss_type = "JAC"  # or "JAC" "MSE" "Devito"
    num_sample = 10
    offset = 414
    data_type = "NS" # "Darcy" "NS"
    
    
    # Load configuration and dataset. and checkpoint
    if data_type == "Darcy":
        if loss_type == "JAC" and num_vec == 50:
            config = load_config("configs/eigenvectors/e_50.yaml")
            ckpt_path = f"checkpoints/n=400_e=50_m=FNO_s=RFS_l=JAC_20250624_120949/n=400_e=50_m=FNO_s=RFS_l=JAC_epoch=187_val_rel_l2_loss=0.0172.ckpt"
        elif loss_type == "JAC" and num_vec == 200:
            config = load_config("configs/eigenvectors/e_200.yaml")
            ckpt_path = f"checkpoints/n=400_e=200_m=FNO_s=RFS_l=JAC_20250615_133916/n=400_e=200_m=FNO_s=RFS_l=JAC_epoch=190_val_rel_l2_loss=0.0170.ckpt"
        elif loss_type == "JAC" and num_vec == 400:
            config = load_config("configs/eigenvectors/e_400.yaml")
            ckpt_path = f"checkpoints/n=400_e=400_m=FNO_s=RFS_l=JAC_20250617_131205/n=400_e=400_m=FNO_s=RFS_l=JAC_epoch=204_val_rel_l2_loss=0.0172.ckpt"
        elif loss_type == "MSE":
            config = load_config("configs/darcy_MSE.yaml")
            # ckpt_path = "checkpoints/DARCY_MSE/Darcy_training_epoch=249_val_rel_l2_loss=0.0009_MSE_May14.ckpt"
            ckpt_path = "checkpoints/Darcy_training_20250615_133632/Darcy_training_epoch=123_val_rel_l2_loss=0.0173.ckpt"
    elif data_type == "NS":
        if loss_type == "JAC" and num_vec == 200:
            config = load_config("configs/eigenvectors/e_400_NS_new.yaml")
            ckpt_path = f"checkpoints/n=1000_e=200_m=FNO_s=RFS_l=JAC_20250903_013609/n=1000_e=200_m=FNO_s=RFS_l=JAC_epoch=499_val_rel_l2_loss=0.1618.ckpt"
        elif loss_type == "JAC" and num_vec == 400:
            config = load_config("configs/eigenvectors/e_400_NS_new.yaml")
            # ckpt_path = f"checkpoints/n=400_e=400_m=FNO_s=RFS_l=JAC_20250907_142507/n=400_e=400_m=FNO_s=RFS_l=JAC_epoch=183_val_rel_l2_loss=0.1652.ckpt"
            ckpt_path = "checkpoints/n=400_e=400_m=FNO_s=RFS_l=JAC_20250907_142507/n=400_e=400_m=FNO_s=RFS_l=JAC_epoch=170_val_rel_l2_loss=0.1650.ckpt"
        elif loss_type == "MSE":
            config = load_config("configs/eigenvectors/e_0_NS_new.yaml")
            # ckpt_path = "checkpoints/n=1000_m=FNO_l=L2_20250902_101350/n=1000_m=FNO_l=L2_epoch=399_val_rel_l2_loss=0.2206.ckpt"
            ckpt_path = "checkpoints/n=1000_m=FNO_l=L2_20250915_114551/n=1000_m=FNO_l=L2_epoch=122_val_rel_l2_loss=0.1655.ckpt"

    
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
    if data_type == "Darcy":
        data_config = load_config("configs/eigenvectors/e_400.yaml")
        data_config.data_settings.batch_size = 20
    else:
        data_config = load_config("configs/eigenvectors/e_400_NS_new.yaml")
    dataset = get_dataset(data_config.experiment.dataset_type, data_config.data_settings)
    dataloader = dataset.get_dataloader(offset=offset, limit=num_sample)

    # Save RNG state
    with open("rng_state_devito.pkl", "wb") as f:
        pickle.dump({
            "np_random_state": np.random.get_state(),
            "random_state": random.getstate()
        }, f)


    # Initialize a list to hold loss and metric data for each sample.
    loss_data_all = []
    metrics_all = []
    sample_counter = 0
    final_ssim_list = []
    final_l2_list = []

    # Prepare CSV accumulators:
    loss_data_all = []
    sample_counter = 0

    print("Running forward experiment to evaluate surrogate model...")
    compute_forward_and_gradient_errors(model, dataloader, device)


    # Get a single sample from the dataloader to use as input `x`
    # batch = next(iter(dataloader))
    # x = batch['x'].detach()

    # rank, S = estimate_jacobian_rank(model, x, num_probes=3005, tol=1e-10)  # Set tol low to keep full spectrum

    # # Choose your target ranks
    # target_ranks = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000]

    # # Sort S in decreasing order (largest singular values first)
    # S_sorted = torch.sort(S, descending=True).values

    # # Print tolerance needed for each target rank
    # print(f"{'Target Rank':>12} | {'Min Tolerance':>15}")
    # print("-" * 30)
    # for r in target_ranks:
    #     if r <= len(S_sorted):
    #         tol_r = S_sorted[r - 1].item()
    #         print(f"{r:12d} | {tol_r:15.4e}")
    #     else:
    #         print(f"{r:12d} | {'(exceeds dim)':>15}")


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