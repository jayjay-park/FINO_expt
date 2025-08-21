#!/usr/bin/env python3
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURE
# ─────────────────────────────────────────────────────────────────────────────
folder      = "prior_mean_noise(0.01)_ood(+5)"
initial     = "prior_mean"
gd_type     = "_GD"
max_iter    = 1000     # number of iterations in your H5
every       = 100
expt_name   = "input_outdist"
dest_folder = "prior_mean_noise(0.01)_ood(+5)"
os.makedirs(dest_folder, exist_ok=True)

# HDF5 paths for outputs (dataset key 'u')
true_out_file = f"{folder}/inversion_history_output_Devito_{initial}{gd_type}.h5"
surrogate_out = {
    r"FINO ($r = 50$)" : f"{folder}/inversion_history_output_JAC_50_{initial}{gd_type}.h5",
    r"FINO ($r = 200$)": f"{folder}/inversion_history_output_JAC_200_{initial}{gd_type}.h5",
    r"FINO ($r = 400$)": f"{folder}/inversion_history_output_JAC_400_{initial}{gd_type}.h5",
    "MSE-FNO"     : f"{folder}/inversion_history_output_MSE_{initial}{gd_type}.h5",
}
dataset_key = "u"
sample_idx  = 0  # which sample

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD TRUE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
def load_output(path):
    with h5py.File(path, 'r') as f:
        U = f[dataset_key][:]     # (samples, iters, H, W)
    s, t, h, w = U.shape
    return U.reshape(s, t, h*w)  # → (samples, iters, d)

U_true = load_output(true_out_file)[sample_idx, :max_iter]  # (max_iter, d)

# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPUTE RELATIVE L2 ERROR
# ─────────────────────────────────────────────────────────────────────────────
rel2_err = {}

for name, path in surrogate_out.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file {path}")
    U_sur = load_output(path)[sample_idx, :max_iter]  # (max_iter, d)

    errs = np.zeros(max_iter)
    for t in range(max_iter):
        diff = U_sur[t] - U_true[t]
        errs[t] = np.linalg.norm(diff) / (np.linalg.norm(U_true[t]) + 1e-16)
        if (t > 998) or (t == 10): 
            print(name, t, "err: ", errs[t])
    rel2_err[name] = errs

# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE CSV
# ─────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame({"iteration": np.arange(max_iter), **rel2_err})
csv_path = os.path.join(dest_folder, "output_rel2_error_all_methods.csv")
df.to_csv(csv_path, index=False)
print("Wrote:", csv_path)

# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOT
# ─────────────────────────────────────────────────────────────────────────────
cb = plt.get_cmap('Dark2').colors
styles = {
    r"FINO ($r = 50$)" : dict(color=cb[0], marker='o', linestyle='-.'),
    r"FINO ($r = 200$)": dict(color=cb[1], marker='s', linestyle='-'),
    r"FINO ($r = 400$)": dict(color=cb[2], marker='^', linestyle=':'),
    "MSE-FNO"     : dict(color=cb[3], marker='d', linestyle='--'),
}

iters = np.arange(max_iter)

plt.figure(figsize=(7,4))
for name, errs in rel2_err.items():
    plt.semilogy(iters, errs, markevery=every, label=name, **styles[name])

plt.xlabel("Iteration")
plt.ylabel(r'$\frac{\|\mathbf{u}_{\mathrm{sur}}^{(i)} - \mathbf{u}_{\mathrm{true}}^{(i)}\|_2}{\|\mathbf{u}_{\mathrm{true}}^{(i)}\|_2}$')
plt.title(f"Output relative $L_2$ error", fontweight='bold', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, which='both', linestyle='-')
plt.tight_layout()

png_path = os.path.join(dest_folder, f"output_rel2_error_all_{expt_name}{gd_type}.png")
plt.savefig(png_path, dpi=150)
plt.close()
print("Saved plot →", png_path)


# FINO ($r = 50$) 10 err:  0.027338977499928
# FINO ($r = 50$) 999 err:  0.011116448764149078
# FINO ($r = 200$) 10 err:  0.013973536555656006
# FINO ($r = 200$) 999 err:  0.012607881706471
# FINO ($r = 400$) 10 err:  0.012231838582326375
# FINO ($r = 400$) 999 err:  0.008298389303793638
# MSE-FNO 10 err:  0.03018636354380606
# MSE-FNO 999 err:  9.395242129636268