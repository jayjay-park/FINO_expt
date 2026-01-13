#!/usr/bin/env python3
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURE PATHS & PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
folder      = "prior_mean_noise(0.01)_tau(3)_partial(0.25)_correct"
initial     = "prior_mean"#'prior_mean'
gd_type     = '_GD'
max_iter    = 800           # number of iterations in your H5 files
every       = 100            # how often to draw markers
expt_name   = "input_indist"
dest_folder = folder
os.makedirs(dest_folder, exist_ok=True)

# gradient files
# true_grad_file = f"{folder}/inversion_history_gradient_NS_Devito_{initial}{gd_type}.h5"
true_grad_file = f"{folder}/inversion_history_gradient_Devito_{initial}{gd_type}.h5"
# surrogate_files = {
#     r"FINO ($r = 50$)" : f"{folder}/inversion_history_gradient_JAC_50_{initial}{gd_type}.h5",
#     r"FINO ($r = 200$)": f"{folder}/inversion_history_gradient_JAC_200_{initial}{gd_type}.h5",
#     r"FINO ($r = 400$)": f"{folder}/inversion_history_gradient_JAC_400_{initial}{gd_type}.h5",
#     "MSE-FNO"     : f"{folder}/inversion_history_gradient_MSE_{initial}{gd_type}.h5"
# }
surrogate_files = {
        # r"FINO ($r = 50$)" : f"{folder}/inversion_history_gradient_NS_JAC_50_{initial}{gd_type}.h5",
        # r"FINO ($r = 200$)": f"{folder}/inversion_history_gradient_NS_JAC_200_{initial}{gd_type}.h5",
        r"FINO ($r = 400$)": f"{folder}/inversion_history_gradient_NS_JAC_400_{initial}{gd_type}.h5",
        "MSE-FNO"     : f"{folder}/inversion_history_gradient_NS_MSE_{initial}{gd_type}_Devito.h5",
}
dataset_key = "g"
sample_idx  = 0

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD TRUE GRADIENTS
# ─────────────────────────────────────────────────────────────────────────────
def load_grad(path):
    with h5py.File(path, 'r') as f:
        G = f[dataset_key][:]    # shape (samples, iters, H, W)
    s, t, h, w = G.shape
    return G.reshape(s, t, h*w) # → (samples, iters, d)

G_true = load_grad(true_grad_file)[sample_idx, :max_iter]  # (max_iter, d)

# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPUTE COSINE ALIGNMENT & RELATIVE L2 DIFFERENCE
# ─────────────────────────────────────────────────────────────────────────────
cos_align = {}
rel2_diff = {}

for name, path in surrogate_files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}")
    G_sur = load_grad(path)[sample_idx, :max_iter]  # (max_iter, d)

    cosines = np.zeros(max_iter)
    rel2s   = np.zeros(max_iter)
    for t in range(max_iter):
        g_t = G_true[t]
        g_s = G_sur[t]
        dot = np.dot(g_t, g_s)
        n1  = np.linalg.norm(g_t)
        n2  = np.linalg.norm(g_s)
        cosines[t] = dot / (n1 * n2 + 1e-16)
        rel2s[t]   = np.linalg.norm(g_s.flatten() - g_t.flatten(), ord=2) #/ (n1 + 1e-16)

    cos_align[name] = cosines
    rel2_diff[name] = rel2s

# Convert cosine alignment to angles (degrees), handling edge cases
angles_deg = {}
for name, cosines in cos_align.items():
    # clip for numerical safety and map to [0, 180] degrees
    clipped = np.clip(cosines, -1.0, 1.0)
    theta = np.degrees(np.arccos(clipped))
    # If either gradient norm was ~0, your cosines may be NaN/inf earlier; keep NaNs to avoid misleading values
    angles_deg[name] = theta


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE CSVs
# ─────────────────────────────────────────────────────────────────────────────
df_align = pd.DataFrame({"iteration": np.arange(max_iter), **cos_align})
df_align.to_csv(os.path.join(dest_folder, "gradient_alignment_all_methods.csv"), index=False)

df_rel2 = pd.DataFrame({"iteration": np.arange(max_iter), **rel2_diff})
df_rel2.to_csv(os.path.join(dest_folder, "gradient_rel2diff_all_methods.csv"), index=False)

print("Saved CSVs to", dest_folder)

# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOT BOTH FIGURES
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size':       14,
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

# define a consistent style
cb = plt.get_cmap('Dark2').colors
styles = {
    r"FINO ($r = 50$)" : dict(color=cb[0], marker='o', linestyle='-.'),
    r"FINO ($r = 200$)": dict(color=cb[1], marker='s', linestyle='-'),
    r"FINO ($r = 400$)": dict(color=cb[2], marker='^', linestyle=':'),
    "MSE-FNO"     : dict(color=cb[3], marker='d', linestyle='--'),
}

iters = np.arange(max_iter)

# a) Angle alignment (degrees)
# plt.figure(figsize=(7,4))
# for name, thetas in angles_deg.items():
#     plt.plot(iters, thetas, markevery=every, label=name, **styles[name])
# plt.xlabel("Iteration", fontsize=14)
# plt.ylabel(r'$\theta^{(i)}$ (deg)', fontsize=14)
# plt.title("Angular Deviation in Gradient", fontweight='bold', fontsize=16)
# plt.ylim(0, 30)  # angles live in [0, 180]
# plt.legend(fontsize=14)
# plt.grid(True, linestyle='-')
# plt.tight_layout()
# plt.savefig(os.path.join(dest_folder, f"gradient_angle_all_{expt_name}{gd_type}.png"), dpi=150)
# plt.close()

plt.figure(figsize=(7,4))
for name, thetas in angles_deg.items():
    plt.plot(iters, thetas, markevery=every, label=name, **styles[name])

plt.xlabel("Iteration", fontsize=14)
plt.ylabel(r'$\theta^{(i)}$ (deg)', fontsize=14)
plt.title("Angular Deviation in Gradient", fontweight='bold', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, linestyle='-')

# add padding below 0
ymin = -0.5   # or -1.0 for more space
ymax = max(10, np.nanmax([v.max() for v in angles_deg.values()]) * 1.05)
plt.ylim(ymin, ymax)

plt.tight_layout()
plt.savefig(os.path.join(dest_folder, f"gradient_angle_all_{expt_name}{gd_type}.png"), dpi=150)
plt.close()





# b) Relative L2 difference
plt.figure(figsize=(7,4))
for name, rel2s in rel2_diff.items():
    plt.semilogy(iters, rel2s, markevery=every, label=name, **styles[name])
plt.xlabel("Iteration")
plt.ylabel(r'$\frac{\|g_{\mathrm{model}}^{(i)}-g_{\mathrm{NS}}^{(i)}\|_2}{\|g_{\mathrm{NS}}^{(i)}\|_2}$')
plt.title(f"Gradient Relative $L_2$ Error", fontweight='bold', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, which='both', linestyle='-')
plt.tight_layout()
plt.savefig(os.path.join(dest_folder, f"gradient_rel2diff_all_{expt_name}{gd_type}.png"), dpi=150)
plt.close()

print("Saved plots to", dest_folder)


