#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURE
# ─────────────────────────────────────────────────────────────────────────────
folder    = "prior_mean_noise(0.01)_tau(3)_correct"
initial   = "prior_mean"
gd_type   = "_GD"
expt_name = "input_outdist"
dest_folder = folder
os.makedirs(dest_folder, exist_ok=True)

paths = {
    "Numerical Simulator": f"{folder}/loss_statistics_multiple_samples_Devito_{initial}{gd_type}.csv",
    "PINO":                f"{folder}/loss_statistics_multiple_samples_PINO_{initial}{gd_type}.csv",
    "MSE-FNO":             f"{folder}/loss_statistics_multiple_samples_MSE_{initial}{gd_type}.csv",
    r"FINO ($r = 400$)":    f"{folder}/loss_statistics_multiple_samples_JAC_400_{initial}{gd_type}.csv",
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
dfs = {name: pd.read_csv(path) for name, path in paths.items()}

# ─────────────────────────────────────────────────────────────────────────────
# 3. STYLE CONFIG (from plot_ood)
# ─────────────────────────────────────────────────────────────────────────────
pal = sns.color_palette("husl", len(paths))
method_list = list(paths.keys())

symbols = {
    "MSE-FNO":              dict(marker="d", linestyle="--"),
    "Numerical Simulator":  dict(marker="v", linestyle=":"),
    "PINO":                 dict(marker="s", linestyle="-."), 
    r"FINO ($r = 400$)":    dict(marker="^", linestyle=":"),
}
styles = {m: dict(color=pal[i], **symbols[m]) for i, m in enumerate(method_list)}

plt.rcParams.update({
    'font.size':       14,
    'lines.linewidth': 2,
    'lines.markersize': 9,
})

# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOT (MSE vs elapsed time with ribbons)
# ─────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8,6))
for name, df in dfs.items():
    grouped = df.groupby("iteration").agg({"elapsed_s":"mean","inversion_MSE":["mean","std"]})
    mean_elapsed = grouped["elapsed_s"]["mean"].values
    mean_mse = grouped["inversion_MSE"]["mean"].values
    std_mse = grouped["inversion_MSE"]["std"].values

    # --- NEW: truncate at min point ---
    min_idx = np.argmin(mean_mse)
    mean_elapsed = mean_elapsed[:min_idx+1]
    mean_mse     = mean_mse[:min_idx+1]
    std_mse      = std_mse[:min_idx+1]
    
    style = styles[name]
    plt.semilogx(mean_elapsed, mean_mse, label=name, 
             markevery=50, marker=style["marker"],
             linestyle=style["linestyle"], color=style["color"])
    plt.fill_between(mean_elapsed, np.maximum(mean_mse-std_mse,1e-12), 
                     mean_mse+std_mse, alpha=0.3, color=style["color"])

plt.yscale("log")
plt.xlabel("Elapsed time (s)")
plt.ylabel("Model Error in MSE")
plt.title("Inversion error (MSE) vs Runtime", fontweight="bold", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", linestyle="-")
plt.tight_layout()

outpath = os.path.join(dest_folder, f"inversion_MSE_vs_runtime{expt_name}{gd_type}.png")
plt.savefig(outpath, dpi=150)
plt.close()
print("Saved plot →", outpath)
