import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------- Configuration -----------------
# Files assumed to be in 'plot/' folder
FILES = {
    "Darcy_Sim": "plot/singular_values_darcy.csv",
    "NS_Sim":    "plot/singular_values_ns.csv",
    "Darcy_FINO": "plot/svd_nn_Darcy.csv",
    "NS_FINO":    "plot/svd_nn_NS.csv",
    "Darcy_MSE":  "plot/svd_nn_Darcy_mse.csv",
    "NS_MSE":     "plot/svd_nn_NS_mse.csv"
}

STYLES = {
    "Darcy": {"color": "tab:blue", "linestyle": "-"},
    "NS":    {"color": "tab:orange", "linestyle": "-"},
    "MSE":   {"linestyle": "--", "alpha": 0.7},
    "FINO":  {"linestyle": "-", "linewidth": 2},
    "Sim":   {"linestyle": ":", "linewidth": 1.5, "alpha": 0.8} # Dotted line for Truth in comparison
}

NOISE_FLOOR = 1e-2  # 0.01

def get_effective_rank(filepath, threshold):
    """Finds the first index where the mean singular value drops below threshold."""
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath, index_col=0)
        mean_curve = df.mean(axis=1)
        below = mean_curve[mean_curve < threshold]
        if not below.empty:
            return below.index[0]
        return np.inf  # Indicates it never drops below threshold
    except:
        return None

def load_and_plot(ax, filepath, label, color, linestyle='-', alpha_line=1.0, linewidth=2):
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return
    
    try:
        df = pd.read_csv(filepath, index_col=0)
        # Aggregate 3 samples: Mean + Std Dev
        mean = df.mean(axis=1)
        std = df.std(axis=1)
        x = df.index.astype(int)

        ax.plot(x, mean, label=label, color=color, linestyle=linestyle, 
                linewidth=linewidth, alpha=alpha_line)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

# ----------------- Main Plotting -----------------
# sharey=True ensures both plots use the same Y-axis scale
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# --- Calculate Null Space Start Indices ---
darcy_rank = get_effective_rank(FILES["Darcy_Sim"], NOISE_FLOOR)
ns_rank = get_effective_rank(FILES["NS_Sim"], NOISE_FLOOR)

# ==========================================
# PLOT 1: Ground Truth Simulators Only
# ==========================================
load_and_plot(ax1, FILES["NS_Sim"], "NS Simulator", STYLES["NS"]["color"])
load_and_plot(ax1, FILES["Darcy_Sim"], "Darcy Simulator", STYLES["Darcy"]["color"])

# Annotations (Null Space Lines)
ax1.axhline(y=NOISE_FLOOR, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1.text(0, NOISE_FLOOR * 0.8, f"Noise Floor ({NOISE_FLOOR})", color='red', fontsize=8, va='top')

if darcy_rank and darcy_rank != np.inf:
    ax1.axvline(x=darcy_rank, color=STYLES["Darcy"]["color"], linestyle=':', linewidth=2)
    ax1.text(darcy_rank + 5, NOISE_FLOOR * 2, "Null Space\n(Darcy)", color=STYLES["Darcy"]["color"], fontsize=10)

if ns_rank == np.inf:
    ax1.text(0.5, 0.1, "NS Null Space > Max Index", transform=ax1.transAxes, color=STYLES["NS"]["color"], ha='center')
elif ns_rank:
    ax1.axvline(x=ns_rank, color=STYLES["NS"]["color"], linestyle=':', linewidth=2)
    ax1.text(ns_rank + 5, NOISE_FLOOR * 2, "Null Space\n(NS)", color=STYLES["NS"]["color"], fontsize=10)

ax1.set_yscale('log')
ax1.set_title("Ground Truth Simulators")
ax1.set_xlabel(r"Index $i$")
ax1.set_ylabel(r"Singular Value $\sigma_i$")
ax1.grid(True, which="both", ls="-", alpha=0.2)
ax1.legend(loc='upper right')

# ==========================================
# PLOT 2: Models + Ground Truth Comparison
# ==========================================
# 1. Plot Ground Truth (faintly/dotted) for reference
load_and_plot(ax2, FILES["NS_Sim"], "NS (Truth)", STYLES["NS"]["color"], 
              linestyle=STYLES["Sim"]["linestyle"], alpha_line=STYLES["Sim"]["alpha"])
load_and_plot(ax2, FILES["Darcy_Sim"], "Darcy (Truth)", STYLES["Darcy"]["color"], 
              linestyle=STYLES["Sim"]["linestyle"], alpha_line=STYLES["Sim"]["alpha"])

# 2. Plot Models
load_and_plot(ax2, FILES["NS_FINO"], "NS (FINO)", STYLES["NS"]["color"], linestyle="-")
load_and_plot(ax2, FILES["NS_MSE"], "NS (MSE)", STYLES["NS"]["color"], linestyle="--")

load_and_plot(ax2, FILES["Darcy_FINO"], "Darcy (FINO)", STYLES["Darcy"]["color"], linestyle="-")
load_and_plot(ax2, FILES["Darcy_MSE"], "Darcy (MSE)", STYLES["Darcy"]["color"], linestyle="--")

# Annotate Null Space (Same as Plot 1 for consistency)
ax2.axhline(y=NOISE_FLOOR, color='red', linestyle='--', linewidth=1, alpha=0.5)
if darcy_rank and darcy_rank != np.inf:
    ax2.axvline(x=darcy_rank, color='gray', linestyle=':', linewidth=2)

ax2.set_yscale('log')
ax2.set_title("Models vs. Ground Truth")
ax2.set_xlabel(r"Index $i$")
ax2.grid(True, which="both", ls="-", alpha=0.2)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig("plot/svd_comparison_final.png", dpi=300)
plt.show()
print("Plot saved to 'svd_comparison_final.png'")


def create_simulator_plot_single():
    # Create a single plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Calculate Null Space Positions
    darcy_rank = get_effective_rank(FILES["Darcy_Sim"], NOISE_FLOOR)
    ns_rank = get_effective_rank(FILES["NS_Sim"], NOISE_FLOOR)

    # 2. Plot Data
    load_and_plot(ax, FILES["NS_Sim"], "Laminar Flow", STYLES["NS"]["color"])
    load_and_plot(ax, FILES["Darcy_Sim"], "Darcy", STYLES["Darcy"]["color"])

    # 3. Add Annotations
    # Noise Floor Line
    ax.axhline(y=NOISE_FLOOR, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(0, NOISE_FLOOR * 0.8, f"Noise Floor ({NOISE_FLOOR})", color='red', fontsize=14, va='top')

    # Darcy Null Space Line
    if darcy_rank and darcy_rank != np.inf:
        ax.axvline(x=darcy_rank, color=STYLES["Darcy"]["color"], linestyle=':', linewidth=4)
        ax.text(darcy_rank + 5, NOISE_FLOOR * 2, "Null Space\n(Darcy)", 
                color=STYLES["Darcy"]["color"], fontsize=14, fontweight='bold')

    # NS Null Space Line (likely off-chart, but we check)
    if ns_rank == np.inf:
        ax.text(0.5, 0.1, "NS Null Space > Max Index", transform=ax.transAxes, 
                color=STYLES["NS"]["color"], ha='center', fontsize=10)
    elif ns_rank:
        ax.axvline(x=ns_rank, color=STYLES["NS"]["color"], linestyle=':', linewidth=4)
        ax.text(ns_rank + 5, NOISE_FLOOR * 2, "Null Space\n(Laminar Flow)", 
                color=STYLES["NS"]["color"], fontsize=14, fontweight='bold')

    # 4. Formatting
    ax.set_yscale('log')
    ax.set_title("Singular value decay of Jacobian of the Solution Operator", fontsize=14)
    ax.set_xlabel(r"Index $i$", fontsize=12)
    ax.set_ylabel(r"Singular Value $\sigma_i$", fontsize=12)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    
    # Save Output
    output_filename = "plot/svd_simulators_only.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to {output_filename}")
    plt.show()


create_simulator_plot_single()