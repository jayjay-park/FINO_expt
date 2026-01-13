#!/usr/bin/env python3
"""
Compute simulator natural gradient across the full FINO trajectory (Darcy/Devito),
and compare alignment vs FINO gradient.

- Uses your original fisher_approx_vjp_batched() logic (Devito branch),
  corrected for (i,j)->linear indexing and chunk assembly.
- Builds low-rank Fisher sketch Q via Devito adjoint J^T v (compute_gradient_single).
- Computes natural gradient via Woodbury and plots metrics vs iteration.

Outputs:
  - cosine_vs_iter_darcy.png
  - norms_vs_iter_darcy.png
"""

import os
import h5py
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils_plot import *

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIG
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folder     = "noise(0.01)_tau(3)_indist_partial(0.25)_correct_lr=0.01"
# folder     = "smooth_Darcy_noise(0.01)_tau(3)_partial(0.25)_correct"_lr=0.01
initial    = "prior_mean"
gd_type    = "_GD"
sample_idx = 0
subspace = True

# HDF5 paths (adjust if your filenames differ)
path_fino_iter = f"{folder}/inversion_history_JAC_400_{initial}{gd_type}.h5"                # key 'a' : (S,T,H,W)
path_sim_grad  = f"{folder}/inversion_history_gradient_NS_Devito_{initial}{gd_type}_JAC_400.h5" # key 'g' : (S,T,H,W)
path_fino_grad = f"{folder}/inversion_history_gradient_JAC_400_{initial}{gd_type}.h5"       # key 'g' : (S,T,H,W)

# Observation / Fisher sketch params
sigma_obs  = 0.01         # data noise std for Σ^{-1/2} (whitening)
rank_Q     = 400           # number of Fisher sketch columns (J^T v)
chunk_size = 100           # how many probes per Devito-threaded batch
full_obs   = False     # set False to use sparse obs given by (i,j)

# Sparse observation indices (used only if full_obs == False).
# Must be 1-D tensors with same length m; i=row (0..H-1), j=col (0..W-1)
i = None
j = None

# Forcing term and grid size (adjust if you load forcing elsewhere)
H = W = 128
forcing_term = torch.zeros(H, W, device=device, dtype=torch.float32)

# ─────────────────────────────────────────────────────────────────────────────
# IMPORT YOUR DEVITO WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────
# GroundwaterModel must expose:
#   - eval_fwd_op(f_np, x_np, return_array=False)  # forward field for fixed x
#   - compute_gradient_single(model, x_np, probe2d, p_fwd) # returns J^T probe (flattened)
#
# If you already have compute_gradient_single in a utilities file, import it.
from groundwater.devito_op import GroundwaterModel, GroundwaterLayer, GroundwaterEquation
from groundwater.utils import GaussianRandomField, plot_fields

groundwater_model = GroundwaterEquation(H)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def compute_gradient_single(simulator, x_np, probe_2d, p_fwd):
    """
    CPU worker: runs one Devito gradient (J^T v) and returns a flat array.
    """
    g2d = simulator.compute_gradient(x_np, probe_2d, p_fwd)  # [H,W]
    return g2d.reshape(-1)  # → (p,)

def load_h5_array(path, key, s):
    with h5py.File(path, "r") as f:
        arr = f[key][s]  # shape (T,H,W)
    return torch.tensor(arr, dtype=torch.float32, device=device)

def woodbury_natgrad(g_hw, Q, lam=1e-3):
    """
    g_hw: [H,W] (simulator gradient at iterate)
    Q:    [p, r] Fisher sketch with columns J^T Σ^{-1/2} v_k
    return: (λI + QQᵀ)^(-1) g reshaped to [H,W]
    """
    g_flat = g_hw.flatten()
    Qt = Q.T                             # [r,p]
    B  = Qt @ Q                          # [r,r]
    rhs= Qt @ g_flat                     # [r]
    I  = torch.eye(B.shape[0], device=Q.device, dtype=Q.dtype)
    M  = I + (B / lam)
    w  = torch.linalg.solve(M, rhs / lam)
    g_nat_flat = (g_flat - Q @ w) / lam
    return g_nat_flat.view_as(g_hw)

def cosine(a, b):
    a, b = a.flatten(), b.flatten()
    return float((a @ b) / (a.norm() * b.norm()))

def proj_cosine_E(g1, g2, E):
    """
    Cosine between g1 and g2 restricted to the subspace span(E).
    E: [p,k] with orthonormal columns (QR if unsure).
    """
    g1f = g1.flatten()
    g2f = g2.flatten()
    P1 = E @ (E.t() @ g1f)
    P2 = E @ (E.t() @ g2f)
    num = (P1 @ P2).item()
    den = (P1.norm() * P2.norm()).item()
    return num / den

def fisher_cosine_Q(g1, g2, Q):
    """
    Fisher-metric cosine: <g1,g2>_F / (||g1||_F ||g2||_F) with F≈QQ^T.
    Implemented as cosine between Q^T g1 and Q^T g2.
    """
    g1f = g1.flatten()
    g2f = g2.flatten()
    a = Q.t() @ g1f   # [r]
    b = Q.t() @ g2f   # [r]
    num = (a @ b).item()
    den = (a.norm() * b.norm()).item() + 1e-12
    return num / den


# ─────────────────────────────────────────────────────────────────────────────
# FISHER SKETCH (DEVITO) — CORRECTED VERSION
# ─────────────────────────────────────────────────────────────────────────────
def fisher_approx_vjp_batched(model_or_simulator,
                              x0,        # [H,W] tensor
                              L,         # 1-D flattened obs indices (ignored if full_obs=True)
                              sigma,     # observation noise std
                              rank=200, chunk_size=32,
                              loss_type="Devito",
                              forcing_term=None,
                              noise_std=0.01,
                              full_obs=True):
    """
    Returns Q ∈ R^{p×r_eff}, columns ~ J^T Σ^{-1/2} v_k for Devito forward.
    Uses Devito adjoint via compute_gradient_single().
    """
    assert loss_type == "Devito", "Only Devito branch implemented here."

    device = x0.device
    H, W = x0.shape[-2], x0.shape[-1]
    p = H * W

    # Determine obs size m and build probes V_full ∈ R^{m×r_req}
    if full_obs:
        m = p
        L = torch.arange(p, device=device, dtype=torch.long)
    else:
        m = int(L.numel())

    r_req = min(rank, m)  # never ask for more columns than m
    scale = (1.0 / sigma) if noise_std != 0 else 1.0
    V_full = torch.randn(m, r_req, device=device) * scale

    # Orthonormalize probes in obs space (improves conditioning)
    V_full, _ = torch.linalg.qr(V_full)        # => (m, r_eff) with r_eff = min(m, r_req)
    r_eff = V_full.shape[1]                    # actual number of columns after QR

    # Prepare result buffer Q ∈ R^{p×r_eff}
    Q = torch.empty(p, r_eff, device=device)

    # Devito forward (NumPy) at x0
    x_np = x0.detach().cpu().numpy().squeeze()  # [H,W]
    f_np = forcing_term.detach().cpu().numpy() if torch.is_tensor(forcing_term) else forcing_term
    p_fwd = model_or_simulator.eval_fwd_op(f_np, x_np, return_array=False)
    L_np = L.detach().cpu().numpy()

    # Chunk over the ACTUAL number of columns r_eff
    for start in range(0, r_eff, chunk_size):
        end = min(start + chunk_size, r_eff)
        V_chunk = V_full[:, start:end]                 # [m, chunk_len]
        V_np = V_chunk.detach().cpu().numpy()
        chunk_len = V_np.shape[1]                      # actual #cols in this chunk

        # Build probe_flat ∈ R^{chunk_len×p}
        probe_flat = np.zeros((chunk_len, p), dtype=np.float32)
        if full_obs:
            # Full obs: weights align with flattened field
            probe_flat[:, :] = V_np.T
        else:
            # Sparse obs: scatter into observed entries L
            for k in range(chunk_len):
                probe_flat[k, L_np] = V_np[:, k]

        # Run Devito adjoint per probe (threaded)
        grads_np = np.empty((chunk_len, p), dtype=np.float32)
        with ThreadPoolExecutor() as exe:
            futures = {
                exe.submit(
                    compute_gradient_single,
                    model_or_simulator,
                    x_np,
                    probe_flat[k].reshape(H, W),
                    p_fwd
                ): k
                for k in range(chunk_len)
            }
            for fut in as_completed(futures):
                k = futures[fut]
                grads_np[k] = fut.result()

        # Store columns into Q at the matching width
        Q[:, start:start+chunk_len] = torch.from_numpy(grads_np).to(device).T

        # cleanup
        del grads_np, probe_flat
        torch.cuda.empty_cache()

    return Q


# ─────────────────────────────────────────────────────────────────────────────
# LOAD TRAJECTORIES
# ─────────────────────────────────────────────────────────────────────────────
A_fino = load_h5_array(path_fino_iter, "a", sample_idx)   # (T,H,W)
G_sim  = load_h5_array(path_sim_grad,  "g", sample_idx)   # (T,H,W)
G_fino = load_h5_array(path_fino_grad, "g", sample_idx)   # (T,H,W)
T = A_fino.shape[0]
iters = np.arange(T)
print(f"Loaded trajectories with {T} iterations, grid {A_fino.shape[-2:]}")

# load observation indices
with h5py.File(f"{folder}/grf_sample_data_Darcy_0.h5", "r") as f:
    L = torch.tensor(f["L"][:], dtype=torch.long, device=device)
print("Loaded L with", len(L), "indices")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
cos_sim, cos_nat = [], []
norm_sim, norm_nat, norm_fino = [], [], []
cos_proj_sim_list, cos_proj_nat_list, cos_fisher_list = [], [], []

# for t in range(T):
for t in range(0, T + 1, 500):
    x_t      = A_fino[t].to(device)    # [H,W]
    g_sim_t  = G_sim[t].to(device)     # [H,W]
    g_fino_t = G_fino[t].to(device)    # [H,W]

    # Fisher sketch with Devito adjoint @ FINO iterate
    Q = fisher_approx_vjp_batched(
            groundwater_model,
            x_t,
            L,
            sigma=sigma_obs,
            rank=rank_Q,
            chunk_size=chunk_size,
            loss_type="Devito",
            forcing_term=forcing_term,
            noise_std=(sigma_obs if sigma_obs > 0 else 0.0),
            full_obs=full_obs
    )
    if subspace:
        # 1) Simulator-eigen subspace (top-400 from current Q)
        U, S, Vh = torch.linalg.svd(Q, full_matrices=False)  # U: [p, r_eff], S descending
        k_keep = min(400, U.shape[1])
        E_sim = U[:, :k_keep].contiguous()                   # orthonormal columns

        # 2) Cosine restricted to span(E_sim)  (direction-only in top-k)
        cos_proj_sim = proj_cosine_E(g_fino_t, g_sim_t, E_sim)

        # Optional: "direction-only" cosine for the natural gradient in the same subspace
        # (this isolates whether F^{-1} mostly rotates within top-k)
        # NOTE: compute g_nat_t before using this (or move after woodbury block).
        # cos_proj_nat = proj_cosine_E(g_fino_t, g_nat_t, E_sim)

        # 3) Fisher-metric cosine (span(Q), Fisher-weighted)
        cos_fisher = fisher_cosine_Q(g_fino_t, g_sim_t, Q)

        print(f"[{t:03d}] cos_proj_sim(top-400)={cos_proj_sim:.3f}  cos_fisher={cos_fisher:.3f}")

        # === Natural gradient via Woodbury ===
        g_nat_t = woodbury_natgrad(g_sim_t, Q, lam=1e-4)

        # (optional) compare NGD in the same top-400 subspace
        cos_proj_nat = proj_cosine_E(g_fino_t, g_nat_t, E_sim)
        print(f"[{t:03d}] cos_proj_nat(top-400)={cos_proj_nat:.3f}")

        # (optional) store the restricted cosines too for plotting later
        cos_proj_sim_list.append(cos_proj_sim)
        cos_proj_nat_list.append(cos_proj_nat)
        cos_fisher_list.append(cos_fisher)
    else:
        s = torch.linalg.svdvals(Q)
        print("Singular values range:", s.max().item(), s.min().item(), "effective rank =", (s > 1e-10).sum().item())
        print("cond", torch.linalg.cond(Q.T @ Q))

        # Natural gradient via Woodbury
        g_nat_t = woodbury_natgrad(g_sim_t, Q, lam=1e-4)
        plot_triplet(g_sim_t, g_nat_t, g_fino_t, t, 128, "FINO", folder)

        # Metrics
        cos_sim.append(cosine(g_fino_t, g_sim_t))
        cos_nat.append(cosine(g_fino_t, g_nat_t))
        norm_sim.append(float(g_sim_t.norm()))
        norm_nat.append(float(g_nat_t.norm()))
        norm_fino.append(float(g_fino_t.norm()))

        print(f"[{t:03d}] cos(FINO,sim)={cos_sim[-1]:.3f}   cos(FINO,nat)={cos_nat[-1]:.3f}")
        plot_triplet(g_sim_t, g_nat_t, g_fino_t, t, 128, "FINO", folder)




# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.size': 14, 'lines.linewidth': 2})

if subspace:
    df = pd.DataFrame([cos_proj_sim_list, cos_proj_nat_list, cos_fisher_list])
    csv_path = os.path.join(folder, "cos_subspace.csv")
else:
    df = pd.DataFrame([cos_sim, cos_nat])
    csv_path = os.path.join(folder, "cos.csv")
df.to_csv(csv_path, index=False)
print("Wrote:", csv_path)




# Cosine similarity vs iteration
plt.figure(figsize=(8,5))

recorded_iters = np.linspace(0, T, len(cos_proj_sim_list))
cos_proj_sim_interp = np.interp(iters, recorded_iters, cos_proj_sim_list)
cos_proj_nat_interp = np.interp(iters, recorded_iters, cos_proj_nat_list)

if subspace:
    plt.plot(iters, cos_proj_sim_interp, "--d", label=r"$\cos(g_{\mathrm{FINO}},\, g_{\mathrm{sim}})$")
    plt.plot(iters, cos_proj_nat_interp, "-^",  label=r"$\cos(g_{\mathrm{FINO}},\, F^{-1} g_{\mathrm{sim}})$")
else:
    plt.plot(iters, cos_sim, "--d", label=r"$\cos(g_{\mathrm{FINO}},\, g_{\mathrm{sim}})$")
    plt.plot(iters, cos_nat, "-^",  label=r"$\cos(g_{\mathrm{FINO}},\, F^{-1} g_{\mathrm{sim}})$")
plt.xlabel("Iteration"); plt.ylabel("Cosine similarity")
plt.title("Alignment over FINO inversion trajectory (Darcy)")
plt.legend(); plt.tight_layout()
if subspace:
    plt.savefig(f"{folder}/cosine_vs_iter_darcy_subspace.png", dpi=150); plt.close()
else:
    plt.savefig(f"{folder}/cosine_vs_iter_darcy.png", dpi=150); plt.close()

# Gradient norms vs iteration
# plt.figure(figsize=(8,5))
# plt.plot(iters, norm_sim,  ":",  label=r"$\|g_{\mathrm{sim}}\|$")
# plt.plot(iters, norm_nat,  "--", label=r"$\|F^{-1} g_{\mathrm{sim}}\|$")
# plt.plot(iters, norm_fino, "-",  label=r"$\|g_{\mathrm{FINO}}\|$")
# plt.xlabel("Iteration"); plt.ylabel("Gradient norm")
# plt.title("Gradient magnitudes vs iteration (Darcy)")
# plt.legend(); plt.tight_layout()
# plt.savefig(f"{folder}/norms_vs_iter_darcy.png", dpi=150); plt.close()

print("Saved → cosine_vs_iter_darcy.png, norms_vs_iter_darcy.png")