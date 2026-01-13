import h5py
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from utils_plot import load_a, plot_single, plot_single_contour
from skimage.metrics import structural_similarity as ssim

def find_best_iter_ssim(fname, sample_idx, x_true):
    with h5py.File(fname, "r") as f:
        A = f["a"][sample_idx]  # shape (T, H, W)

    x_true_np = x_true.cpu().squeeze().numpy()
    best_idx, best_score = None, -1.0

    for t in range(A.shape[0]):
        a_t = A[t]  # (H, W)
        # binarize
        a_t = binarize_from_phys(torch.tensor(a_t), method="kmeans").cpu().numpy()
        # compute SSIM
        score = ssim(x_true_np, a_t, data_range=a_t.max() - a_t.min())
        if score > best_score:
            best_idx, best_score = t, score

    return best_idx, best_score


# --- helpers (copied from inversion.py) ---
def darcy_mask_tanh(x, a_min=0.1, a_max=0.9):
    return 0.5 * (a_max - a_min) * torch.tanh(x) + 0.5 * (a_max + a_min)

def _kmeans2_threshold(x_phys, iters=10, a_min=0.1, a_max=0.9):
    v = x_phys.detach().flatten()
    c1 = torch.quantile(v, 0.2)
    c2 = torch.quantile(v, 0.8)
    for _ in range(iters):
        d1, d2 = (v - c1).abs(), (v - c2).abs()
        m1, m2 = v[d1 <= d2], v[d2 < d1]
        if len(m1) == 0 or len(m2) == 0:
            break
        c1_new, c2_new = m1.mean(), m2.mean()
        if torch.isclose(c1, c1_new) and torch.isclose(c2, c2_new):
            break
        c1, c2 = c1_new, c2_new
    return 0.5 * (c1 + c2)

def binarize_from_phys(x_phys, method="kmeans", a_min=0.1, a_max=0.9):
    if method == "kmeans":
        mid_val = _kmeans2_threshold(x_phys)
    elif method == "median":
        mid_val = torch.median(x_phys)
    else:
        mid_val = 0.5 * (a_min + a_max)
    return torch.where(x_phys >= mid_val, 
                       torch.tensor(a_max, device=x_phys.device, dtype=x_phys.dtype),
                       torch.tensor(a_min, device=x_phys.device, dtype=x_phys.dtype))

from scipy.ndimage import generic_filter

def majority_filter(binary_array, size=3):
    return generic_filter(binary_array, 
                          lambda x: np.round(np.mean(x)), 
                          size=size)

# # --- main ---
# if __name__ == "__main__":
#     # path to your inversion history file
#     opt_type = "Devito"
#     folder = "prior_mean_noise(0.05)_tau(3)"
#     out_folder = "prior_mean_noise(0.05)_tau(3)"
#     os.makedirs(out_folder, exist_ok=True)

#     sample_idx, iter_idx = 0, 900   # choose sample and iteration
#     fname = f"{folder}/inversion_history_{opt_type}_prior_mean_GD.h5"
#     fname_true = f"{folder}/grf_sample_data_{sample_idx}.h5"
    

#     with h5py.File(fname_true, "r") as f:
#         field = f["x"][sample_idx]   # shape (H, W)
#         x_true = torch.tensor(field)
#         print("x_true", x_true.shape)

#     with h5py.File(fname, "r") as f:
#         field = f["a"][sample_idx, iter_idx]   # shape (H, W)
#         x = torch.tensor(field)

#     # map to physical range (if stored as raw logits, otherwise skip)
#     x_phys = darcy_mask_tanh(x)

#     # binarize
#     x_bin = binarize_from_phys(x_phys, method="median")
#     # x_bin = majority_filter(x_bin).reshape(128, 128)

def find_best_iter(fname, sample_idx, x_true):
    with h5py.File(fname, "r") as f:
        A = f["a"][sample_idx]  # shape (T, H, W)

    x_true = x_true.to(torch.float32).flatten()
    best_idx, best_err = None, float("inf")

    for t in range(A.shape[0]):  # loop over iterations
        a_t = torch.tensor(A[t], dtype=torch.float32).flatten()
        a_t =  binarize_from_phys(a_t, method="kmeans")
        mse = torch.mean((a_t - x_true) ** 2).item()
        if mse < best_err:
            best_idx, best_err = t, mse

    return best_idx, best_err

# --- main ---
if __name__ == "__main__":
    opt_type = "Devito_Darcy"#"JAC_Darcy_200" #"Devito_Darcy"
    folder = "."#"prior_mean_noise(0.01)_tau(3)_partial(0.25)_correct"
    out_folder = "."#"prior_mean_noise(0.01)_tau(3)_partial(0.25)_correct"
    os.makedirs(out_folder, exist_ok=True)

    sample_idx = 0 #7
    fname = f"{folder}/inversion_history_{opt_type}_prior_mean_GD.h5"
    fname_true = f"grf_sample_data_Darcy_prior_mean_{sample_idx}.h5"

    # load true field
    with h5py.File(fname_true, "r") as f:
        field = f["x"][0]   # shape (H, W)
        x_true = torch.tensor(field)

    # find best iteration
    # best_iter, best_err = find_best_iter(fname, sample_idx, x_true)
    # print(f"Best iteration = {best_iter}, MSE = {best_err:.4e}")
    # best_iter = 1000
    
    # find best iteration by SSIM
    # best_iter_ssim, best_ssim = find_best_iter_ssim(fname, sample_idx, x_true)
    # print(f"Best iteration by SSIM = {best_iter_ssim}, SSIM = {best_ssim:.4f}")

    # load corresponding iterate
    with h5py.File(fname, "r") as f:
        field = f["a"][0]  # shape (H, W)
        x = torch.tensor(field)

    iteration = 100
    x_exp = torch.exp(x[iteration, :, :])
    x_plain = x[iteration, :,:]
    print("exp", x_exp.shape)

    # plot
    print(x_exp.min(), x_exp.max())
    plot_single(x_true.squeeze().cpu().numpy(), f"{out_folder}/sanity_check_groundt_x.png")
    # plot_single(x_bin.cpu().numpy(), f"{out_folder}/binary_sample_{opt_type}_{sample_idx}_iter{best_iter}.png", vmin=0.1, vmax=0.9)
    plot_single_contour(x_exp.cpu().numpy(), f"{out_folder}/perm_{opt_type}_{sample_idx}_iter{iteration}.png", vmin=0.1, vmax=3.2)
    plot_single(x_exp.cpu().numpy(), f"{out_folder}/perm_{opt_type}_{sample_idx}_iter{iteration}_.png", vmin=-1, vmax=2.5)

    print(f"Saved binary plot for sample {sample_idx}, iter {best_iter}")

# PINO
# Best iteration = 472, MSE = 7.7070e-02
# Best iteration by SSIM = 292, SSIM = 0.5850
# JAC400
# Best iteration = 392, MSE = 6.5234e-02
# Best iteration by SSIM = 193, SSIM = 0.5820
# Saved binary plot for sample 7, iter 392