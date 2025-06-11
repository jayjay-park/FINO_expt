import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics_comparison(csv_jac_10, csv_jac_50, csv_jac_100, csv_mse, csv_rand, metrics, color_map, output_filename, metric_name):
    """
    Loads CSV files (JAC variants, MSE, RAND), computes mean and std over time,
    and plots each metric on a separate subplot.
    """
    # Load data
    df_jac_10 = pd.read_csv(csv_jac_10)
    df_jac_50 = pd.read_csv(csv_jac_50)
    df_jac_100 = pd.read_csv(csv_jac_100)
    df_mse = pd.read_csv(csv_mse)
    df_rand = pd.read_csv(csv_rand)

    # Layout
    n_metrics = len(metrics)
    ncols = 2
    nrows = (n_metrics + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Group by 'time' for time-based plotting
        grouped_jac_10 = df_jac_10.groupby("elapsed_s")[metric].agg(["mean", "std"]).reset_index()
        grouped_jac_50 = df_jac_50.groupby("elapsed_s")[metric].agg(["mean", "std"]).reset_index()
        grouped_jac_100 = df_jac_100.groupby("elapsed_s")[metric].agg(["mean", "std"]).reset_index()
        grouped_mse = df_mse.groupby("elapsed_s")[metric].agg(["mean", "std"]).reset_index()
        grouped_rand = df_rand.groupby("elapsed_s")[metric].agg(["mean", "std"]).reset_index()

        color_jac, color_mse, color_rand = color_map.get(metric, ("blue", "orange", "gray"))

        # Plot JAC variants
        ax.plot(grouped_jac_10["elapsed_s"], grouped_jac_10["mean"],
                label="Jvp:FIM (10)", color=color_jac, linestyle='--', linewidth=2)
        ax.fill_between(grouped_jac_10["elapsed_s"],
                        grouped_jac_10["mean"] - 2 * grouped_jac_10["std"],
                        grouped_jac_10["mean"] + 2 * grouped_jac_10["std"],
                        color=color_jac, alpha=0.2)

        ax.plot(grouped_jac_50["elapsed_s"], grouped_jac_50["mean"],
                label="Jvp:FIM (50)", color=color_jac, linestyle='-', linewidth=2)
        ax.fill_between(grouped_jac_50["elapsed_s"],
                        grouped_jac_50["mean"] - 2 * grouped_jac_50["std"],
                        grouped_jac_50["mean"] + 2 * grouped_jac_50["std"],
                        color=color_jac, alpha=0.2)

        ax.plot(grouped_jac_100["elapsed_s"], grouped_jac_100["mean"],
                label="Jvp:FIM (100)", color=color_jac, linestyle='-.', linewidth=2)
        ax.fill_between(grouped_jac_100["elapsed_s"],
                        grouped_jac_100["mean"] - 2 * grouped_jac_100["std"],
                        grouped_jac_100["mean"] + 2 * grouped_jac_100["std"],
                        color=color_jac, alpha=0.2)

        # Plot MSE
        ax.plot(grouped_mse["elapsed_s"], grouped_mse["mean"],
                label="MSE", color=color_mse, marker='s', markevery=200, linewidth=2)
        ax.fill_between(grouped_mse["elapsed_s"],
                        grouped_mse["mean"] - 2 * grouped_mse["std"],
                        grouped_mse["mean"] + 2 * grouped_mse["std"],
                        color=color_mse, alpha=0.2)

        # Plot RAND
        ax.plot(grouped_rand["elapsed_s"], grouped_rand["mean"],
                label="Numerical Simulator", color=color_rand, marker='^', markevery=200, linewidth=2)
        ax.fill_between(grouped_rand["elapsed_s"],
                        grouped_rand["mean"] - 2 * grouped_rand["std"],
                        grouped_rand["mean"] + 2 * grouped_rand["std"],
                        color=color_rand, alpha=0.2)

        ax.set_xlabel("elapsed_s", fontsize=12)
        ax.set_ylabel(metric_name[idx], fontsize=12)
        ax.set_title(f"{metric_name[idx]}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True)

    # Clean up extra axes
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    plt.show()


# File paths
csv_jac_10 = "metrics_per_minute_JAC_10_smooth.csv"
csv_jac_50 = "metrics_per_minute_JAC_50_smooth.csv"
csv_jac_100 = "metrics_per_minute_JAC_100_smooth.csv"
csv_mse = "metrics_per_minute_MSE_smooth.csv"
csv_rand = "metrics_per_minute_Devito_smooth.csv"

# Metrics to plot
metrics = ["loss", "inversion_MSE", "regularization", "SSIM"]
metric_name = ["Data Residual", r"Inversion Metric: $\frac{\|\mathbf{a}^\ast - \mathbf{a}\|_2}{\|\mathbf{a^\ast}\|_2}$", "Regularization Term (Laplacian)", "Inversion Metric: SSIM"]
index_name = ["MSE", r"$\frac{\|\mathbf{a}^\ast - \mathbf{a}\|_2}{\|\mathbf{a^\ast}\|_2}$", "Regularization Term (Laplacian)", "Inversion Metric: SSIM"]


# Color map for each metric (JAC, MSE, RAND)
color_map = {
    "loss": ("blueviolet", "red", "blue"), #("#1f77b4", "#aec7e8", "#7f7f7f"),
    "inversion_MSE": ("blueviolet", "red", "blue"), #("#ff7f0e", "#ffbb78", "#c7c7c7"),
    "regularization": ("blueviolet", "red", "blue"), # ("#2ca02c", "#98df8a", "#bcbd22"),
    "SSIM": ("blueviolet", "red", "blue") #("#d62728", "#ff9896", "#8c564b")
}

# Global tick size
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Generate plot
plot_metrics_comparison(
    csv_jac_10, csv_jac_50, csv_jac_100,
    csv_mse, csv_rand,
    metrics, color_map,
    "metrics_comparison_over_time.png", metric_name
)
