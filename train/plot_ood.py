#!/usr/bin/env python3
import os
import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.ns_inversion import NSModel  # Your model
from utils import get_dataset, load_config, get_model  # Your utils
from test_forward import compute_Jvp
from utils_plot import *
import seaborn as sns

def compute_jvp_errors(model, iterates, v, true_jvps):
    """
    model      : trained surrogate model
    iterates   : (T, H, W) or (S, T, H, W) array of iterates
    v          : probe vector(s) for Jvp (same shape as x)
    true_jvps  : list/array of simulator Jvp for each iterate

    Returns: (T,) array of L2 misfits
    """
    errs = []
    for t in range(iterates.shape[1]):  # loop over iterations
        x_t = torch.tensor(iterates[0, t]).unsqueeze(0).unsqueeze(0).float().to(v.device)
        _, jvp_pred = compute_Jvp(model, x_t, v)
        jvp_true = true_jvps[t]
        diff = (jvp_pred - jvp_true).flatten()
        errs.append(torch.norm(diff).item())
    return np.array(errs)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURE
# ─────────────────────────────────────────────────────────────────────────────
folder      = "."#"noise(0.01)_tau(3)_prior_mean_partial(0.25)_correct_lr=0.01"#"prior_mean_noise(0.01)_tau(3)_correct"#"smooth_Darcy_noise(0.01)_tau(3)_partial(0.25)_correct"
dest_folder = folder
initial     = "prior_mean"
gd_type     = "_GD"
max_iter    = 1500     # number of iterations in your H5
every_iter  = 100
every       = 100
expt_name   = "input_indist"
term_type = "gradient" # output
full_grad   = False
os.makedirs(dest_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
print(f"Using device: {device}")
# Define simulation parameters.
num_vec = 400
num_sample = 100
offset = 414
data_type = "Darcy" # "Darcy" "NS"
dim       = 128

# HDF5 paths for outputs (dataset key 'u')
true_out_file = f"{folder}/inversion_history_{term_type}_Devito_{data_type}_{initial}{gd_type}.h5"
if term_type == "gradient":
    simulator_grad = {
    # noise(0.01)_tau(3)_smooth_partial(0.25)_lr=0.01/inversion_history_gradient_NS_Devito_Darcy_smooth_GD_MSE_Darcy.h5
    # noise(0.01)_tau(3)_smooth_partial(0.25)_lr=0.01/inversion_history_gradient_NS_Devito_smooth_GD_MSE_Darcy.h5
    "MSE-FNO"     : f"{folder}/inversion_history_gradient_NS_Devito_{data_type}_{initial}{gd_type}_MSE_{data_type}.h5",
    # 'PINO':       f'{folder}/inversion_history_gradient_NS_Devito_{initial}{gd_type}_PINO.h5',
    r"FINO ($r = 400$)": f"{folder}/inversion_history_gradient_NS_Devito_{data_type}_{initial}{gd_type}_JAC_{data_type}_400.h5",
    }

    surrogate_grad = {
    "MSE-FNO"     : f"{folder}/inversion_history_gradient_MSE_{data_type}_{initial}{gd_type}.h5",
    # 'PINO':       f'{folder}/inversion_history_gradient_PINO_{initial}{gd_type}.h5',
    r"FINO ($r = 400$)": f"{folder}/inversion_history_gradient_JAC_{data_type}_400_{initial}{gd_type}.h5",
    }
    dataset_key = "g"
else:
    surrogate_out = {
        r"FINO ($r = 50$)" : f"{folder}/inversion_history_{term_type}_JAC_50_{initial}{gd_type}.h5",
        r"FINO ($r = 200$)": f"{folder}/inversion_history_{term_type}_JAC_200_{initial}{gd_type}.h5",
        r"FINO ($r = 400$)": f"{folder}/inversion_history_{term_type}_JAC_400_{initial}{gd_type}.h5",
        "MSE-FNO"     : f"{folder}/inversion_history_{term_type}_MSE_{initial}{gd_type}.h5",
    }
    dataset_key = "u"
sample_idx  = 0  # which sample

# Load dataset
if data_type == "Darcy":
    data_config = load_config("configs/eigenvectors/e_400.yaml")
    data_config.data_settings.batch_size = 20
else:
    data_config = load_config("configs/eigenvectors/e_200_NS_new.yaml")
dataset = get_dataset(data_config.experiment.dataset_type, data_config.data_settings)
dataloader = dataset.get_dataloader(offset=offset, limit=num_sample)
v = None
for batch in dataloader:
    v = batch['v'][0].detach().cpu().numpy().reshape(dim*dim, -1)
# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD TRUE/SURROGATE OUTPUTS (vectorized across samples)
# ─────────────────────────────────────────────────────────────────────────────
# U_true_all = load_a(true_out_file, dataset_key)
# S, T, D = U_true_all.shape
# T_use = min(max_iter, T)
# print("T_use", T_use, S, T, D)
T_use = 1500


# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPUTE PER-ITERATION RELATIVE L2 ERROR (across samples)
#    err_s(t) = ||U_sur[s,t]-U_true[s,t]||_2 / (||U_true[s,t]||_2 + eps)
#    Then mean_t = mean_s err_s(t), std_t = std_s err_s(t)
# ─────────────────────────────────────────────────────────────────────────────
eps = 0
rel2_stats = {}  # name -> dict(mean=..., std=..., per_sample=...)
rel2_stats_jvp = {}

for name, path in simulator_grad.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file {path}")
    print("name", name)
    print(path)
    U_sim_all = load_a(path, dataset_key)
    U_sur_all = load_a(surrogate_grad[name], dataset_key)

    print(U_sim_all.shape)

    # truncate to T_use
    t_idx = np.arange(1, T_use * every_iter, every_iter)    # [1, 101, 201, ...]
    t_idx = t_idx[t_idx < U_sur_all.shape[1]]  # safety
    U_t = U_sim_all[:, :T_use, :]           # (S, T_use, D)
    V_t = U_sur_all[:, t_idx, :]            # (S, T_use, D)
    print("U_t", U_t.shape, "V_t", V_t.shape)


    # Example calls
    plot_fields_triplet(U_t, V_t, idx=0, dim=dim, name=name, dest_folder=dest_folder)
    plot_fields_triplet(U_t, V_t, idx=1, dim=dim, name=name, dest_folder=dest_folder)
    plot_fields_triplet(U_t, V_t, idx=2, dim=dim, name=name, dest_folder=dest_folder)
    plot_fields_triplet(U_t, V_t, idx=5, dim=dim, name=name, dest_folder=dest_folder)
    # plot_fields_triplet(U_t, V_t, idx=19, dim=dim, name=name, dest_folder=dest_folder)
    # plot_fields_triplet(U_t, V_t, idx=24, dim=dim, name=name, dest_folder=dest_folder)
    # plot_fields_triplet(U_t, V_t, idx=30, dim=dim, name=name, dest_folder=dest_folder)
    # plot_fields_triplet(U_t, V_t, idx=39, dim=dim, name=name, dest_folder=dest_folder)


    if full_grad:
        # errs = np.linalg.norm(U_t - V_t, axis=-1)
        # print("errs", errs)

        # per-sample, per-iter L2 norms
        # num_sur = V_t / np.linalg.norm(V_t, axis=-1, keepdims=True)           # (S, T_use)
        # num_true = U_t / np.linalg.norm(U_t, axis=-1, keepdims=True)           # (S, T_use)
        # errs = np.linalg.norm(num_sur - num_true, axis=-1)

        num = np.linalg.norm(V_t - U_t, axis=-1)
        den = np.linalg.norm(U_t, axis=-1) + eps           # (S, T_use)
        errs = num / den                                   # (S, T_use)
        # print("num", num)
        # print("den", den)
        # print("min", U_t[0,0,:].min(), U_t[0,0,:].max())

        # per-sample, per-iter MSE
        # sq_errs = (V_t - U_t) ** 2                   # (S, T_use, D)
        # errs = sq_errs.mean(axis=-1)                  # (S, T_use) average over D
    else:
        # Project gradients onto the top-r eigenspace
        U_proj = U_t @ v              # (S, T, r)
        V_proj = V_t @ v              # (S, T, r)
        print("U_proj", U_proj.shape)

        # Compute relative L2 error in the projected subspace
        num = np.linalg.norm(V_proj - U_proj, axis=-1)
        den = np.linalg.norm(U_proj, axis=-1)
        print("num", num)
        # print("den", den)
        errs = num / den


    rel2_stats[name] = {
        "mean": errs.mean(axis=0),                     # (T_use,)
        "std":  errs.std(axis=0),                      # (T_use,)
        "per_sample": errs,                            # (S, T_use) if you need it
    }

    # ─────────────────────────────────────────────────────────────
    # Compute cosine similarity (directional alignment)
    # ─────────────────────────────────────────────────────────────

    # Optional: project onto top-r Fisher subspace (if V_r available)
    if full_grad:
        U_proj, V_proj = U_t, V_t
    else:
        U_proj = U_t @ v       # (S, T_use, r)
        V_proj = V_t @ v       # (S, T_use, r)
        print("Sanity check", U_t.shape, v.shape, U_proj.shape)
        # Sanity check (10, 40, 16384) (16384, 400) (10, 40, 400)

    # Compute cosine similarity per sample/iteration
    num = np.sum(V_proj * U_proj, axis=-1)                # numerator ⟨g_nn, g_true⟩
    den = np.linalg.norm(V_proj, axis=-1) * np.linalg.norm(U_proj, axis=-1)
    cosine = num / den                                    # (S, T_use)

    # Save stats
    rel2_stats[name]["cos_mean"] = cosine.mean(axis=0)
    rel2_stats[name]["cos_std"]  = cosine.std(axis=0)


    eps = 1e-12
    U = U_t[-1, :].reshape(-1)      # simulator gradient field at t*
    V = V_t[-1, :].reshape(-1)      # surrogate gradient field at t*

    def cos(a,b):
        return float((a @ b) / ((np.linalg.norm(a)*np.linalg.norm(b)) + eps))

    # 1) Raw cosine (what you plotted)
    print("cos(full)      =", cos(U, V))

    # 2) Remove DC: if this flips sign -> global offset caused negativity
    U0 = U - U.mean()
    V0 = V - V.mean()
    print("cos(zero-mean) =", cos(U0, V0))

    # Cosine at last iteration only
    U_last = U_t[0, -1, :]
    V_last = V_t[0, -1, :]
    print("cos(last iter) =", cos(U_last, V_last))

    # Cosine over all concatenated iterations (your current print)
    U_all = U_t.reshape(-1)
    V_all = V_t.reshape(-1)
    print("cos(all iters concatenated) =", cos(U_all, V_all))



# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE CSV (mean & std by iteration)
# ─────────────────────────────────────────────────────────────────────────────
# df_cols = {"iteration": np.arange(T_use)}
df_cols = {"iteration": np.arange(0, T_use, every_iter)}
for name, stats in rel2_stats.items():
    df_cols[f"{name}__mean"] = stats["mean"]
    df_cols[f"{name}__std"]  = stats["std"]
df = pd.DataFrame(df_cols)
csv_path = os.path.join(dest_folder, "gradient_rel2_error_all_methods_mean_std.csv")
df.to_csv(csv_path, index=False)
print("Wrote:", csv_path)

# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOT (mean ± std ribbon)
# ─────────────────────────────────────────────────────────────────────────────
pal = sns.color_palette("husl", 4) # pal = sns.color_palette("husl", len(surrogate_grad)+1)
method_list = ["MSE-FNO", "Numerical Simulator", "PINO", r"FINO ($r = 400$)"]#list(surrogate_grad.keys())

symbols = {
    "MSE-FNO":           dict(marker="d", linestyle="--"),
    "Numerical Simulator": dict(marker="v", linestyle=":"),
    "PINO":              dict(marker="s", linestyle="-."),
    r"FINO ($r = 400$)": dict(marker="^", linestyle=":"),
}

styles = {m: dict(color=pal[i], **symbols[m]) for i, m in enumerate(method_list)}
iters_full = np.arange(T_use)
iters = np.arange(0, T_use, every_iter)


plt.rcParams.update({
    'font.size':       14,
    'lines.linewidth': 2,
    'lines.markersize': 10,
})

plt.figure(figsize=(8,5))
# for name, st in rel2_stats.items():
#     m, s = st["mean"], st["std"]
#     print("m sahpe", m.shape, iters.shape)
#     style = styles[name]
#     # plot mean with markers
#     plt.plot(iters.squeeze(), m, label=name, markevery=every, marker=style["marker"],
#                  linestyle=style["linestyle"], color=style["color"])
#     # add shaded band for variance
#     plt.fill_between(iters, np.maximum(m - s, 1e-12), m + s,
#                      alpha=0.3, color=style["color"])
for name, st in rel2_stats.items():
    m, s = st["mean"], st["std"]

    # case 1: simulator gradient saved every 100th step (shorter array)
    if len(m) < len(iters_full):
        recorded_iters = np.linspace(0, T_use - 1, len(m))
        m = np.interp(iters_full, recorded_iters, m)
        s = np.interp(iters_full, recorded_iters, s)

    style = styles[name]
    plt.plot(iters_full, m, label=name, markevery=every_iter,
             marker=style["marker"], linestyle=style["linestyle"], color=style["color"])
    plt.fill_between(iters_full, np.maximum(m - s, 1e-12), m + s, alpha=0.3, color=style["color"])



plt.xlabel("Iteration")
if full_grad:
    # plt.ylabel(r'$\mathbb{E}\left[\|\frac{\mathbf{g}^{nn}(a_i^\text{nn})}{\|\mathbf{g}^{nn}(a_i^\text{nn})\|_2} - \frac{\mathbf{g}(a_i^\text{nn})}{\|\mathbf{g}(a_i^\text{nn})\|_2}\|_2\right]$')
    plt.ylabel(r'$\frac{\|\mathbf{g}^{nn}(a_i^\text{nn})- \mathbf{g}(a_i^\text{nn})\|_2}{\|\mathbf{g}(a_i^\text{nn})\|_2}$')
    # plt.ylabel(r'$\|\mathbf{g}^{nn}(\mathbf{a}_i^\text{nn})- \mathbf{g}(\mathbf{a}_i^\text{nn})\|_2$')
    plt.title(r"$L_2$ misfit of gradient at $\mathbf{a}_i^\text{nn}$", fontweight='bold', fontsize=16)
    plt.title(r"Relative $L_2$ misfit of gradient at $\mathbf{a}_i^\text{nn}$", fontweight='bold', fontsize=16)
    png_path = os.path.join(dest_folder, f"ood_{expt_name}{gd_type}_mean_std_wrtNN_full.png")
else:
    # plt.ylabel(r'$\|V_r(\mathbf{g}^{nn}(\mathbf{a}_i^\text{nn})- \mathbf{g}(\mathbf{a}_i^\text{nn}))\|_2$')
    # plt.title(r"$L_2$ misfit of gradient at $\mathbf{a}_i^\text{nn}$ at Fisher Eigenspace", fontweight='bold', fontsize=16)
    # plt.ylabel(r'$\|V_r^\top(\mathbf{g}^{nn}-\mathbf{g})\|_2$')
    plt.ylabel(r'$\frac{\|V_r^\top\mathbf{g}^{nn}(a_i^\text{nn})- V_r^\top\mathbf{g}(a_i^\text{nn})\|_2}{\|V_r^\top\mathbf{g}(a_i^\text{nn})\|_2}$')
    # plt.title(r"$L_2$ misfit of gradient in top-$r$ Fisher eigenspace", fontweight='bold', fontsize=16)
    plt.title(r"Relative $L_2$ misfit of gradient in top-$r$ Fisher eigenspace", fontweight='bold', fontsize=16)
    png_path = os.path.join(dest_folder, f"ood_{expt_name}{gd_type}_mean_std_wrtNN.png")


plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='-')
plt.tight_layout()
plt.savefig(png_path, dpi=150)
plt.close()
print("Saved plot →", png_path)


# ─────────────────────────────────────────────────────────────
# COSINE SIMILARITY PLOT
# ─────────────────────────────────────────────────────────────
plt.figure(figsize=(8,5))
for name, st in rel2_stats.items():
    if "cos_mean" not in st:
        continue
    m, s = st["cos_mean"], st["cos_std"]

    if len(m) < len(iters_full):
        recorded_iters = np.linspace(0, T_use - 1, len(m))
        m = np.interp(iters_full, recorded_iters, m)
        s = np.interp(iters_full, recorded_iters, s)

    true_iter = np.arange(1, len(m) * every_iter + 1, every_iter)
    style = styles.get(name, dict(color='k', marker='o', linestyle='-'))
    plt.plot(iters_full, m, label=name, markevery=every_iter,
             marker=style["marker"], linestyle=style["linestyle"], color=style["color"])
    plt.fill_between(iters_full, np.clip(m - s, -1, 1), np.clip(m + s, -1, 1),
                     alpha=0.3, color=style["color"])
    # plt.plot(true_iter, m, label=name, markevery=every_iter,
            #  marker=style["marker"], linestyle=style["linestyle"], color=style["color"])
    # plt.fill_between(true_iter, np.clip(m - s, -1, 1), np.clip(m + s, -1, 1),
                    #  alpha=0.3, color=style["color"])


plt.xlabel("Iteration")
if full_grad:
    plt.ylabel(r'$\cos\!\angle(\mathbf{g}^{nn},\,\mathbf{g})$')
    plt.title(r"Cosine similarity of gradient directions", fontweight='bold', fontsize=16)
    cos_png = os.path.join(dest_folder, f"ood_{expt_name}{gd_type}_cosine_similarity_full.png")
else:
    plt.ylabel(r'$\cos\!\angle(V_r^\top\mathbf{g}^{nn},\,V_r^\top\mathbf{g})$')
    plt.title(r"Cosine similarity of gradients in top-$r$ Fisher eigenspace", fontweight='bold', fontsize=16)
    cos_png = os.path.join(dest_folder, f"ood_{expt_name}{gd_type}_cosine_similarity.png")

plt.ylim([-1.05, 1.05])
plt.legend(fontsize=12)
plt.grid(True, linestyle='-')
plt.tight_layout()
plt.savefig(cos_png, dpi=150)
plt.close()
print("Saved cosine similarity plot →", cos_png)
