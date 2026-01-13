#!/usr/bin/env python3
"""
plot_ood_ngd_finoFIM.py

Compute simulator natural gradient across the full FINO trajectory (Darcy/Devito),
compare alignment vs FINO gradient, and ALSO test a FINO-FIM preconditioner built
from FINO's eigenvectors (no eigenvalues needed — we estimate them from the simulator
Fisher sketch Q at each iterate).

Outputs (under `folder/`):
  - grad_triplet_{t}_FINO.png           (triplets with shared colorbar)
  - cosine_vs_iter_darcy.png            (cosines over iterations)
  - norms_vs_iter_darcy.png             (norms over iterations)
  - cos.csv                             (CSV with cos(FINO,sim), cos(FINO,F^-1 g_sim), cos(FINO,P_E g_sim))
"""

import os
import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from models.ns_inversion import NSModel  # Your model
from utils import get_dataset, load_config, get_model  # Your utils
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils_inversion import *
from utils_plot import *

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIG
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folder     = "prior_mean_noise(0.01)_tau(3)_partial(0.25)_correct"
initial    = "prior_mean"
gd_type    = "_GD"
sample_idx = 0
data_type  = "Darcy"
dim        = 128

# HDF5 paths (adjust if your filenames differ)
path_fino_iter = f"{folder}/inversion_history_JAC_400_{initial}{gd_type}.h5"                # key 'a' : (S,T,H,W)
path_sim_grad  = f"{folder}/inversion_history_gradient_NS_Devito_{initial}{gd_type}_JAC_400.h5" # key 'g' : (S,T,H,W)
path_fino_grad = f"{folder}/inversion_history_gradient_JAC_400_{initial}{gd_type}.h5"       # key 'g' : (S,T,H,W)

# Observation / Fisher sketch params
sigma_obs  = 0.01         # data noise std for Σ^{-1/2} (whitening)
rank_Q     = 400          # number of Fisher sketch columns (J^T v)
chunk_size = 100          # how many probes per Devito-threaded batch
full_obs   = False        # set False to use sparse obs given by L

# Grid / forcing
H = W = 128
forcing_term = torch.zeros(H, W, device=device, dtype=torch.float32)

# Observation indices (L is loaded from GRF file)
obs_file = "grf_sample_data_0.h5"     # change if needed
obs_L_key = "L"

# Natural-gradient damping (Woodbury)
lam_nat = 1e-6

# FINO-FIM preconditioner damping (applied with estimated eigenvalues)
lam_damp_finoP = 1e-6

# ─────────────────────────────────────────────────────────────────────────────
# IMPORT DEVITO WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────
# We use the same API as your working code:
#   GroundwaterEquation.eval_fwd_op(...)
#   GroundwaterEquation.compute_gradient(...)
from groundwater.devito_op import GroundwaterEquation

simulator = GroundwaterEquation(H)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def compute_gradient_single(simulator, x_np, probe_2d, p_fwd):
    """CPU worker: one Devito adjoint J^T v → flattened gradient (p,)."""
    g2d = simulator.compute_gradient(x_np, probe_2d, p_fwd)  # [H,W]
    return g2d.reshape(-1)

def load_h5_array(path, key, s):
    with h5py.File(path, "r") as f:
        arr = f[key][s]  # shape (T,H,W)
    return torch.tensor(arr, dtype=torch.float32, device=device)

def cosine(a, b, eps=1e-12):
    a, b = a.flatten(), b.flatten()
    return float((a @ b) / (a.norm() * b.norm() + eps))

def woodbury_natgrad(g_hw, Q, lam=1e-3):
    """
    g_hw: [H,W]
    Q:    [p, r]
    returns (λI + QQᵀ)^(-1) g
    """
    g_flat = g_hw.flatten()
    Qt = Q.T
    B  = Qt @ Q                          # [r,r]
    rhs= Qt @ g_flat                     # [r]
    I  = torch.eye(B.shape[0], device=Q.device, dtype=Q.dtype)
    M  = I + (B / lam)
    w  = torch.linalg.solve(M, rhs / lam)
    g_nat_flat = (g_flat - Q @ w) / lam
    return g_nat_flat.view_as(g_hw)

def plot_triplet_shared(U_t, V_t, W_t, idx, dim, name, dest_folder):
    """Three heatmaps with shared symmetric colorbar."""
    U = U_t.reshape(dim, dim).detach().cpu().numpy()
    V = V_t.reshape(dim, dim).detach().cpu().numpy()
    W = W_t.reshape(dim, dim).detach().cpu().numpy()
    global_max = np.max(np.abs([U, V, W]))
    vmin, vmax = -global_max, global_max

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ims = []
    for arr, ax in zip([U, V, W], axes):
        im = ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax)
        ims.append(im); ax.axis("off")
    titles = [r"$g_{\mathrm{sim}}$", r"$g_{\mathrm{sim}}^{\mathrm{nat}}$", r"$g_{\mathrm{FINO}}$"]
    for ax, t in zip(axes, titles): ax.set_title(t, fontsize=16)
    cbar = fig.colorbar(ims[0], ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("Gradient value", fontsize=12)
    plt.tight_layout()
    os.makedirs(dest_folder, exist_ok=True)
    plt.savefig(f"{dest_folder}/grad_triplet_{idx}_{name}.png", dpi=300)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FISHER SKETCH (Devito adjoint, L-based)
# ─────────────────────────────────────────────────────────────────────────────
def fisher_approx_vjp_batched(sim,
                              x0,        # [H,W]
                              L,         # 1-D flattened obs indices (ignored if full_obs=True)
                              sigma,     # noise std
                              rank=200, chunk_size=32,
                              forcing_term=None,
                              full_obs=True):
    """
    Returns Q ∈ R^{p×r_eff}, columns ≈ J^T Σ^{-1/2} v_k (Devito adjoint).
    """
    device = x0.device
    H, W = x0.shape[-2], x0.shape[-1]
    p = H * W

    if full_obs:
        m = p
        L = torch.arange(p, device=device, dtype=torch.long)
    else:
        m = int(L.numel())

    r_req = min(rank, m)
    scale = 1.0 / sigma
    V_full = torch.randn(m, r_req, device=device) * scale

    # Orthonormalize probes in obs space
    V_full, _ = torch.linalg.qr(V_full)  # (m, r_eff)
    r_eff = V_full.shape[1]

    Q = torch.empty(p, r_eff, device=device)

    # Forward solve
    x_np = x0.detach().cpu().numpy().squeeze()
    f_np = forcing_term.detach().cpu().numpy() if torch.is_tensor(forcing_term) else forcing_term
    p_fwd = sim.eval_fwd_op(f_np, x_np, return_array=False)

    L_np = L.detach().cpu().numpy()

    for start in range(0, r_eff, chunk_size):
        end = min(start + chunk_size, r_eff)
        V_chunk = V_full[:, start:end]        # [m, chunk_len]
        V_np = V_chunk.detach().cpu().numpy()
        chunk_len = V_np.shape[1]

        probe_flat = np.zeros((chunk_len, p), dtype=np.float32)
        if full_obs:
            probe_flat[:, :] = V_np.T
        else:
            for k in range(chunk_len):
                probe_flat[k, L_np] = V_np[:, k]

        grads_np = np.empty((chunk_len, p), dtype=np.float32)
        with ThreadPoolExecutor() as exe:
            futures = {
                exe.submit(
                    compute_gradient_single,
                    sim,
                    x_np,
                    probe_flat[k].reshape(H, W),
                    p_fwd
                ): k
                for k in range(chunk_len)
            }
            for fut in as_completed(futures):
                k = futures[fut]
                grads_np[k] = fut.result()

        Q[:, start:start+chunk_len] = torch.from_numpy(grads_np).to(device).T
        del grads_np, probe_flat
        torch.cuda.empty_cache()

    return Q

# ─────────────────────────────────────────────────────────────────────────────
# FINO-FIM: load eigenvectors, estimate eigenvalues from Q, build preconditioner
# ─────────────────────────────────────────────────────────────────────────────
def load_fino_evecs(path, key="v", device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    # Load dataset
    if data_type == "Darcy":
        data_config = load_config("configs/eigenvectors/e_400.yaml")
        data_config.data_settings.batch_size = 20
    else:
        data_config = load_config("configs/eigenvectors/e_200_NS_new.yaml")
    dataset = get_dataset(data_config.experiment.dataset_type, data_config.data_settings)
    dataloader = dataset.get_dataloader(offset=414, limit=2)
    V = None
    for batch in dataloader:
        V = batch['v'][0].detach().cpu().reshape(dim*dim, -1)
    # Ensure columns are eigenvectors: shape p×k
    if V.ndim != 2:
        raise ValueError(f"Eigenvectors must be 2D; got {V.shape}")
    if V.shape[0] < V.shape[1]:  # if saved as k×p, transpose
        V = V.T
    # Orthonormalize columns
    E, _ = torch.linalg.qr(V, mode="reduced")
    return E.to(device)

def estimate_lambdas_from_sketch(E, Q):
    """
    E: [p,k] (orthonormal columns)
    Q: [p,r] (Fisher sketch columns)
    Returns lam_hat: [k] with lam_i ≈ (1/r) ||(E^T Q)_{i,:}||^2
    """
    R = E.T @ Q                         # [k, r]
    lam_hat = (R.pow(2).sum(dim=1) / max(1, R.shape[1])).clamp_min(1e-12)
    return lam_hat

def apply_fino_precond_with_lams(g_hw, E, lam_hat, lam_damp=1e-2):
    """
    P_E g = E (Lam+λI)^{-1} E^T g + (1/λ) (I - E E^T) g
    """
    g = g_hw.flatten()
    alpha = E.T @ g                                # [k]
    alpha_tilde = alpha / (lam_hat + lam_damp)     # [k]
    g_par  = E @ alpha_tilde                       # [p]
    g_perp = g - E @ alpha                         # [p]
    g_nat  = g_par + (1.0/lam_damp) * g_perp
    return g_nat.view_as(g_hw)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD TRAJECTORIES & OBS
# ─────────────────────────────────────────────────────────────────────────────
A_fino = load_h5_array(path_fino_iter, "a", sample_idx)   # (T,H,W)
G_sim  = load_h5_array(path_sim_grad,  "g", sample_idx)   # (T,H,W)
G_fino = load_h5_array(path_fino_grad, "g", sample_idx)   # (T,H,W)
T = A_fino.shape[0]
iters = np.arange(T)
print(f"Loaded trajectories with {T} iterations, grid {A_fino.shape[-2:]}")

with h5py.File(obs_file, "r") as f:
    L = torch.tensor(f[obs_L_key][:], dtype=torch.long, device=device)
print("Loaded L with", len(L), "indices")

# FINO eigenvectors (only vectors; eigenvalues will be estimated from Q)
path_fino_evecs=None
evecs_key='v'
E = load_fino_evecs(path_fino_evecs, key=evecs_key, device=device)
print("Loaded FINO eigenvectors E with shape:", tuple(E.shape))

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
cos_sim, cos_nat, cos_finoP = [], [], []
norm_sim, norm_nat, norm_fino, norm_finoP = [], [], [], []

for t in range(T):
    x_t      = A_fino[t].to(device)    # [H,W]
    g_sim_t  = G_sim[t].to(device)     # [H,W]
    g_fino_t = G_fino[t].to(device)    # [H,W]

    # Fisher sketch with Devito adjoint @ FINO iterate
    Q = fisher_approx_vjp_batched(
            simulator,
            x_t,
            L=L,
            sigma=sigma_obs,
            rank=rank_Q,
            chunk_size=chunk_size,
            forcing_term=forcing_term,
            full_obs=full_obs
    )

    # Natural gradient via Woodbury
    g_nat_t = woodbury_natgrad(g_sim_t, Q, lam=lam_nat)

    # Estimate FINO-FIM eigenvalues from this Q and apply FINO preconditioner
    lam_hat = estimate_lambdas_from_sketch(E, Q)                # [k]
    g_finoP_t = apply_fino_precond_with_lams(g_sim_t, E, lam_hat, lam_damp=lam_damp_finoP)

    # Triplet plot (sim, nat(sim), FINO) — shared colorbar
    plot_triplet_shared(g_sim_t, g_nat_t, g_fino_t, t, H, "FINO", folder)

    # Metrics
    cos_sim.append(cosine(g_fino_t, g_sim_t))
    cos_nat.append(cosine(g_fino_t, g_nat_t))
    cos_finoP.append(cosine(g_fino_t, g_finoP_t))
    norm_sim.append(float(g_sim_t.norm()))
    norm_nat.append(float(g_nat_t.norm()))
    norm_fino.append(float(g_fino_t.norm()))
    norm_finoP.append(float(g_finoP_t.norm()))

    print(f"[{t:03d}] cos(FINO,sim)={cos_sim[-1]:.3f}   "
          f"cos(FINO,F^-1 g)={cos_nat[-1]:.3f}   cos(FINO,P_E g)={cos_finoP[-1]:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.size': 14, 'lines.linewidth': 2})

# Cosine similarity vs iteration
plt.figure(figsize=(8,5))
plt.plot(iters, cos_sim,  "--d", label=r"$\cos(g_{\mathrm{FINO}},\, g_{\mathrm{sim}})$")
plt.plot(iters, cos_nat,  "-^",  label=r"$\cos(g_{\mathrm{FINO}},\, F^{-1} g_{\mathrm{sim}})$")
plt.plot(iters, cos_finoP,"-o",  label=r"$\cos(g_{\mathrm{FINO}},\, P_E g_{\mathrm{sim}})$")
plt.xlabel("Iteration"); plt.ylabel("Cosine similarity")
plt.title("Alignment over FINO inversion trajectory (Darcy)")
plt.legend(); plt.tight_layout()
plt.savefig(f"{folder}/cosine_vs_iter_darcy_sub.png", dpi=150); plt.close()

# Gradient norms vs iteration
plt.figure(figsize=(8,5))
plt.plot(iters, norm_sim,   ":",  label=r"$\|g_{\mathrm{sim}}\|$")
plt.plot(iters, norm_nat,   "--", label=r"$\|F^{-1} g_{\mathrm{sim}}\|$")
plt.plot(iters, norm_fino,  "-",  label=r"$\|g_{\mathrm{FINO}}\|$")
plt.plot(iters, norm_finoP, "-.", label=r"$\|P_E g_{\mathrm{sim}}\|$")
plt.xlabel("Iteration"); plt.ylabel("Gradient norm")
plt.title("Gradient magnitudes vs iteration (Darcy)")
plt.legend(); plt.tight_layout()
plt.savefig(f"{folder}/norms_vs_iter_darcy_sub.png", dpi=150); plt.close()

# CSV of cosines
df = pd.DataFrame({
    "cos_fino_sim":   cos_sim,
    "cos_fino_nat":   cos_nat,
    "cos_fino_PEg":   cos_finoP,
})
csv_path = os.path.join(folder, "cos_sub.csv")
df.to_csv(csv_path, index=False)
print("Wrote:", csv_path)
print("Saved → cosine_vs_iter_darcy.png, norms_vs_iter_darcy.png")
