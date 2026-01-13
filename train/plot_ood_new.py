#!/usr/bin/env python3
"""
Plot gradient L2 misfit and cosine similarity (corrected indexing & cosine).
- Uses true iteration indices [1, 101, 201, ..., 1 + 100*(T_sim-1)] based on simulator snapshots.
- Aligns surrogate gradients to those exact iterations.
- Computes cosine per-sample & per-iteration with optional zero-mean (DC-bias removal).
- Plots mean±std across samples without incorrect interpolation.

Assumes your H5 files and util functions as in the original script.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from utils import get_dataset, load_config
from utils_plot import *
from test_forward import compute_Jvp  # (import kept for parity; not used below)

# ─────────────────────────────────────────────────────────────────────────────
# User CONFIG (kept from your script; adjust as needed)
# ─────────────────────────────────────────────────────────────────────────────
folder      = "noise(0.01)_tau(3)_smooth(less)_partial(0.25)_lr=0.008"
dest_folder = folder
initial     = "smooth"
gd_type     = "_GD"
dim         = 128
every_iter  = 100           # simulator saved stride
expt_name   = "input_indist"
term_type   = "gradient"    # "gradient" or other
full_grad   = True

# Runtime/data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
num_vec    = 400
num_sample = 100
offset     = 414
data_type  = "Darcy"  # "Darcy" or "NS"
zero_mean_cosine = True  # remove DC offset before cosine

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
if term_type == "gradient":
    simulator_grad = {
        "MSE-FNO":          f"{folder}/inversion_history_gradient_NS_Devito_{data_type}_{initial}{gd_type}_MSE_{data_type}.h5",
        r"FINO ($r = 400$)": f"{folder}/inversion_history_gradient_NS_Devito_{data_type}_{initial}{gd_type}_JAC_{data_type}_400.h5",
    }
    surrogate_grad = {
        "MSE-FNO":          f"{folder}/inversion_history_gradient_MSE_{data_type}_{initial}{gd_type}.h5",
        r"FINO ($r = 400$)": f"{folder}/inversion_history_gradient_JAC_{data_type}_400_{initial}{gd_type}.h5",
    }
    dataset_key = "g"
else:
    raise NotImplementedError("This corrected script is intended for term_type='gradient'.")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_cosine_stats(U_t, V_t, zero_mean=True):
    """U_t, V_t: (S, T, D). Returns cosine_all (S,T), mean (T,), std (T,)"""
    S, T, D = U_t.shape
    cosine_all = np.zeros((S, T), dtype=np.float64)
    for s in range(S):
        for t in range(T):
            u = U_t[s, t]
            v = V_t[s, t]
            if zero_mean:
                u = u - u.mean()
                v = v - v.mean()
            nu = np.linalg.norm(u)
            nv = np.linalg.norm(v)
            if nu == 0 or nv == 0:
                cosine = np.nan
            else:
                cosine = float(np.dot(u, v) / (nu * nv))
            cosine_all[s, t] = cosine
    cos_mean = np.nanmean(cosine_all, axis=0)
    cos_std  = np.nanstd(cosine_all, axis=0)
    return cosine_all, cos_mean, cos_std


# Optional: dataset (kept for parity; only used to fetch v if needed)
if data_type == "Darcy":
    data_config = load_config("configs/eigenvectors/e_400.yaml")
    data_config.data_settings.batch_size = 20
else:
    data_config = load_config("configs/eigenvectors/e_200_NS_new.yaml")
dataset = get_dataset(data_config.experiment.dataset_type, data_config.data_settings)
dataloader = dataset.get_dataloader(offset=offset, limit=num_sample)
# grab one v just to mirror your original environment
v = None
for batch in dataloader:
    v = batch['v'][0].detach().cpu().numpy().reshape(dim*dim, -1)
    break

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
rel2_stats = {}

pal = sns.color_palette("husl", 4)
method_list = ["MSE-FNO", r"FINO ($r = 400$)"]
symbols = {
    "MSE-FNO":            dict(marker="d", linestyle="--"),
    r"FINO ($r = 400$)":   dict(marker="^", linestyle=":"),
}
styles = {m: dict(color=pal[i], **symbols[m]) for i, m in enumerate(method_list)}

plt.rcParams.update({'font.size':14,'lines.linewidth':2,'lines.markersize':10})

for name, sim_path in simulator_grad.items():
    sur_path = surrogate_grad[name]
    if not os.path.exists(sim_path):
        raise FileNotFoundError(f"Missing simulator file {sim_path}")
    if not os.path.exists(sur_path):
        raise FileNotFoundError(f"Missing surrogate file {sur_path}")

    print("name", name)
    print(sim_path)

    U_sim_all = load_a(sim_path, dataset_key)  # (S_sim, T_sim, D)
    U_sur_all = load_a(sur_path, dataset_key)  # (S_sur, T_sur, D)

    # Select samples consistently
    print("shape", U_sim_all.shape)

    S_sim, T_sim, D = U_sim_all.shape
    S_sur, T_sur, _ = U_sur_all.shape
    assert S_sim == S_sur, "Sample count mismatch after slicing"

    # True iteration indices for simulator snapshots (1-based stride of 100)
    true_iters = 1 + np.arange(T_sim) * every_iter   # [1, 101, ..., 1+100*(T_sim-1)]

    # Align surrogate to those iterations (clip any that exceed T_sur-1)
    t_idx = true_iters.copy()
    t_idx[t_idx >= T_sur] = T_sur - 1  # guard: if 4000 exists it's fine; if 4001 would clip

    # Slice aligned tensors
    U_t = U_sim_all[:, :len(true_iters), :]                   # (S, T_sim, D)
    V_t = U_sur_all[:, t_idx, :]                              # (S, T_sim, D)

    print("U_t", U_t.shape, "V_t", V_t.shape)

    # Optional triplet plots (indices within [0, T_sim-1])
    for idx in [0, 1, 2, 5, 20, 30, T_sim-1]:
        if 0 <= idx < T_sim:
            try:
                plot_fields_triplet(U_t, V_t, idx=idx, dim=dim, name=name, dest_folder=dest_folder)
            except Exception as e:
                print(f"Triplet plot failed at idx={idx}: {e}")

    # ---------------- L2 misfit (choose one) ----------------
    if full_grad:
        # Raw L2 (magnitude sensitive)
        errs = np.linalg.norm(V_t - U_t, axis=-1)          # (S, T)
        # Relative L2 (optional)
        # denom = np.linalg.norm(U_t, axis=-1) + 1e-12
        # errs = errs / denom
    else:
        # Projected L2 in Fisher subspace (requires v with shape (D, r))
        U_proj = U_t @ v   # (S, T, r)
        V_proj = V_t @ v   # (S, T, r)
        errs = np.linalg.norm(V_proj - U_proj, axis=-1)

    rel2_stats[name] = {
        "mean": errs.mean(axis=0),
        "std":  errs.std(axis=0),
    }

    # ---------------- Cosine similarity (corrected) ----------------
    cosine_all, cos_mean, cos_std = compute_cosine_stats(U_t, V_t, zero_mean=zero_mean_cosine)
    rel2_stats[name]["cos_mean"] = cos_mean
    rel2_stats[name]["cos_std"]  = cos_std

    # Diagnostics: per-sample final iteration
    print(f"[{name}] per-sample cosine @ last iter:", cosine_all[:, -1])
    print(f"[{name}] mean@last = {cos_mean[-1]:.4f}, std@last = {cos_std[-1]:.4f}")

    # Also print single-sample zero-mean cosine at last iter for sample 0
    u_last = U_t[0, -1] - U_t[0, -1].mean()
    v_last = V_t[0, -1] - V_t[0, -1].mean()
    cos_last0 = float(np.dot(u_last, v_last) / (np.linalg.norm(u_last)*np.linalg.norm(v_last) + 1e-12))
    print(f"[{name}] sample0 zero-mean cosine@last = {cos_last0:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE CSV (mean & std by iteration)
# ─────────────────────────────────────────────────────────────────────────────
csv_df = {"iteration": true_iters}
for name, st in rel2_stats.items():
    csv_df[f"{name}__l2_mean"] = st["mean"]
    csv_df[f"{name}__l2_std"]  = st["std"]
    csv_df[f"{name}__cos_mean"] = st["cos_mean"]
    csv_df[f"{name}__cos_std"]  = st["cos_std"]

df = pd.DataFrame(csv_df)
csv_path = os.path.join(dest_folder, "gradient_rel2_error_all_methods_mean_std_corrected.csv")
df.to_csv(csv_path, index=False)
print("Wrote:", csv_path)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS (no incorrect interpolation)
# ─────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8,5))
for name, st in rel2_stats.items():
    m, s = st["mean"], st["std"]
    style = styles.get(name, dict(color='k', marker='o', linestyle='-'))
    plt.semilogy(true_iters, m, label=name, markevery=len(true_iters),
                 marker=style["marker"], linestyle=style["linestyle"], color=style["color"]) 
    plt.fill_between(true_iters, np.maximum(m - s, 1e-12), m + s, alpha=0.3, color=style["color"])
plt.xlabel("Iteration")
plt.ylabel(r'$\|\mathbf{g}^{nn}-\mathbf{g}\|_2$')
plt.title(r"$L_2$ misfit of gradient at surrogate iterate", fontweight='bold', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='-')
plt.tight_layout()
plt.savefig(os.path.join(dest_folder, f"ood_{expt_name}{gd_type}_mean_std_wrtNN_full_corrected.png"), dpi=150)
plt.close()
print("Saved L2 plot (corrected)")

plt.figure(figsize=(8,5))
for name, st in rel2_stats.items():
    m, s = st["cos_mean"], st["cos_std"]
    style = styles.get(name, dict(color='k', marker='o', linestyle='-'))
    plt.plot(true_iters, m, label=name, markevery=len(true_iters),
             marker=style["marker"], linestyle=style["linestyle"], color=style["color"]) 
    plt.fill_between(true_iters, np.clip(m - s, -1, 1), np.clip(m + s, -1, 1), alpha=0.3, color=style["color"]) 
plt.xlabel("Iteration")
plt.ylabel(r'$\cos(\mathbf{g}^{nn},\,\mathbf{g})$')
plt.ylim([-1.05, 1.05])
plt.title(r"Cosine similarity of gradient directions", fontweight='bold', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='-')
plt.tight_layout()
plt.savefig(os.path.join(dest_folder, f"ood_{expt_name}{gd_type}_cosine_similarity_full_corrected.png"), dpi=150)
plt.close()
print("Saved cosine similarity plot (corrected)")
