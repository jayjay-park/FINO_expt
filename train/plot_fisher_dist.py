import h5py
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from inversion import fisher_approx_vjp_batched
from inversion import GroundwaterModel, GroundwaterEquation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_a_list(h5_path, sample_idx=0):
    """Load list of parameter iterates a_t from HDF5 inversion history file."""
    with h5py.File(h5_path, 'r') as f:
        A = f['a'][sample_idx]  # shape: (num_iters, H, W)
    a_list = [torch.tensor(A[t], dtype=torch.float32, device=device).flatten()
              for t in range(A.shape[0])]
    return a_list

def fisher_step_distance(a_list, Q):
    """Compute per-step and cumulative Fisher distances given low-rank Q."""
    d_F = []
    for t in range(len(a_list) - 1):
        delta = a_list[t+1] - a_list[t]
        proj = torch.matmul(Q.T, delta)  # (r,)
        dist = torch.norm(proj, p=2).item()
        d_F.append(dist)
    d_F = torch.tensor(d_F)
    D_F = torch.cumsum(d_F, dim=0)
    return d_F, D_F

def plot_fisher_distances(d_F_dict, D_F_dict, epsilon, file_path):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    for name, d_F in d_F_dict.items():
        plt.plot(range(len(d_F)), d_F.cpu().numpy(), label=name)
    plt.axhline(y=epsilon, color='k', linestyle='--', label='Trust boundary')
    plt.xlabel("Iteration")
    plt.ylabel("Step Fisher Distance")
    plt.title("Per-step Fisher Distance")
    plt.legend()

    plt.subplot(1,2,2)
    for name, D_F in D_F_dict.items():
        plt.plot(range(len(D_F)), D_F.cpu().numpy(), label=name)
    plt.axhline(y=epsilon, color='k', linestyle='--', label='Trust boundary')
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative Fisher Distance")
    plt.title("Cumulative Fisher Distance")
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(file_path)
    plt.close()

# === Example usage ===
folder = "prior_mean_noise(0.01)_ood(+5)"
methods = {
    "FINO (r=200)": f"{folder}/inversion_history_JAC_200_prior_mean_GD.h5",
    "MSE-FNO": f"{folder}/inversion_history_MSE_prior_mean_GD.h5",
}
file_path = f"{folder}/fisher_dist"

epsilon = 0.2  # trust boundary radius

# You need your simulator object to compute Q
forcing_term = torch.zeros(128, 128, device=device)
gw_torch_model = GroundwaterModel(forcing_term.shape[0])
groundwater_eq = GroundwaterEquation(forcing_term.shape[0])

# Observation indexing (full obs in your default setup)
grid_size = 128
i, j = torch.meshgrid(torch.arange(grid_size, device=device),
                      torch.arange(grid_size, device=device),
                      indexing='ij')
i = i.reshape(-1)
j = j.reshape(-1)

d_F_dict, D_F_dict = {}, {}

for name, path in methods.items():
    a_list = load_a_list(path, sample_idx=0)

    # Compute Q for the first iterate a0 using simulator's FIM approx
    a0 = a_list[0].reshape(1, 1, 128, 128)
    Q = fisher_approx_vjp_batched(
        groundwater_eq, a0, i, j,
        sigma=0.01, rank=200, chunk_size=50, loss_type="Devito",
        forcing_term=forcing_term
    )
    # Optional: orthonormalize Q
    Q, _ = torch.linalg.qr(Q)

    d_F, D_F = fisher_step_distance(a_list, Q)
    d_F_dict[name] = d_F
    D_F_dict[name] = D_F

plot_fisher_distances(d_F_dict, D_F_dict, epsilon, file_path)
