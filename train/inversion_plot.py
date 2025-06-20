import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py


def compute_avg_time(csv_file, max_iter=1250):
    df = pd.read_csv(csv_file)
    df_filtered = df[df["iteration"] < max_iter]
    # Sum elapsed seconds per sample
    per_sample_time = df_filtered.groupby("sample")["elapsed_s"].max()
    return per_sample_time.mean()


def plot_metrics_comparison(csv_jac_1, csv_jac_10, csv_jac_50, csv_jac_100, csv_mse, csv_rand,
                            h5_jac_1, h5_jac_10, h5_jac_50, h5_jac_100, h5_rand, h5_mse,
                            metrics, color_map, output_filename, metric_name, index_name):
    """
    Plot selected metrics with 3 rows: optimization, inversion, and relative norm errors.
    """

    # Load CSVs
    df_jac_1 = pd.read_csv(csv_jac_1)
    df_jac_10 = pd.read_csv(csv_jac_10)
    df_jac_50 = pd.read_csv(csv_jac_50)
    df_jac_100 = pd.read_csv(csv_jac_100)
    df_jac_200 = pd.read_csv(csv_jac_200)
    df_mse = pd.read_csv(csv_mse)
    df_rand = pd.read_csv(csv_rand)

    def compute_rel_norm_per_batch(h5_path_model: str,
                        h5_path_devito: str,
                        p: float = np.inf,
                        max_iter: int = 2200,
                        sample_indices: np.ndarray = None):
        with h5py.File(h5_path_model, 'r') as f_m, h5py.File(h5_path_devito, 'r') as f_d:
            if sample_indices is not None:
                A_m = f_m['a'][sample_indices]  # shape = (S, T, H, W)
                A_d = f_d['a'][sample_indices]
            else:
                A_m = f_m['a'][:]  # fallback to full load (not memory-safe for large data)
                A_d = f_d['a'][:]
        
        S, T, H, W = A_m.shape
        if max_iter is None:
            max_iter = T

        mean_rel = np.zeros(max_iter, dtype=float)
        std_rel = np.zeros(max_iter, dtype=float)

        for t in range(max_iter):
            M_flat = A_m[:, t].reshape(S, -1)
            D_flat = A_d[:, max_iter].reshape(S, -1)
            diff = M_flat - D_flat

            if p == np.inf:
                sample = np.max(np.abs(diff), axis=1)
                den = np.max(np.abs(D_flat), axis=1)
            else:
                sample = np.linalg.norm(diff, ord=p, axis=1)
                den = np.linalg.norm(D_flat, ord=p, axis=1)

            rel_values = sample / den
            mean_rel[t] = np.mean(rel_values)
            std_rel[t] = np.std(rel_values)

        return mean_rel, std_rel

    def compute_rel_norm(h5_path_model, h5_path_devito,
                                   p=np.inf, max_iter=2200,
                                   batch_size=5):
        all_means = []
        all_stds = []

        with h5py.File(h5_path_model, 'r') as f:
            total_samples = f['a'].shape[0]

        for s0 in range(0, total_samples, batch_size):
            s1 = min(s0 + batch_size, total_samples)
            sample_indices = np.arange(s0, s1)
            print(f"Batch: samples {s0}–{s1}", flush=True)

            mean_b, std_b = compute_rel_norm_per_batch(
                h5_path_model, h5_path_devito,
                p=p, max_iter=max_iter,
                sample_indices=sample_indices
            )

            all_means.append(mean_b)
            all_stds.append(std_b)

        all_means = np.stack(all_means)  # shape (num_batches, T)
        all_stds = np.stack(all_stds)

        # Mean and std across all samples
        mean_over_all = np.mean(all_means, axis=0)
        std_over_all = np.sqrt(np.mean(all_stds**2, axis=0))  # pooled std approximation

        return mean_over_all, std_over_all





    print("Compute rel norms")

    # rel_2_1_mean, rel_2_1_std  = compute_rel_norm(h5_jac_1, h5_rand, p=2, max_iter=max_iter)
    # rel_2_10_mean, rel_2_10_std  = compute_rel_norm(h5_jac_10, h5_rand, p=2, max_iter=max_iter)
    # rel_2_50_mean, rel_2_50_std  = compute_rel_norm(h5_jac_50, h5_rand, p=2, max_iter=max_iter)
    # rel_2_100_mean, rel_2_100_std = compute_rel_norm(h5_jac_100, h5_rand, p=2, max_iter=max_iter)
    rel_2_200_mean, rel_2_200_std = compute_rel_norm(h5_jac_200, h5_rand, p=2, max_iter=max_iter)
    rel_2_mse_mean, rel_2_mse_std = compute_rel_norm(h5_mse, h5_rand, p=2, max_iter=max_iter)

    print("Compute rel norms infty")

    # rel_inf_1_mean, rel_inf_1_std   = compute_rel_norm(h5_jac_1, h5_rand, p=np.inf, max_iter=max_iter)
    # rel_inf_10_mean, rel_inf_10_std  = compute_rel_norm(h5_jac_10, h5_rand, p=np.inf, max_iter=max_iter)
    # rel_inf_50_mean, rel_inf_50_std  = compute_rel_norm(h5_jac_50, h5_rand, p=np.inf, max_iter=max_iter)
    # rel_inf_100_mean, rel_inf_100_std = compute_rel_norm(h5_jac_100, h5_rand, p=np.inf, max_iter=max_iter)
    rel_inf_200_mean, rel_inf_200_std = compute_rel_norm(h5_jac_200, h5_rand, p=np.inf, max_iter=max_iter)
    rel_inf_mse_mean, rel_inf_mse_std = compute_rel_norm(h5_mse, h5_rand, p=np.inf, max_iter=max_iter)

    print("Compute avg time")

    # avg_time_jac_1 = compute_avg_time(csv_jac_1, max_iter)
    # avg_time_jac_10 = compute_avg_time(csv_jac_10, max_iter)
    # avg_time_jac_50 = compute_avg_time(csv_jac_50, max_iter)
    # avg_time_jac_100 = compute_avg_time(csv_jac_100, max_iter)
    avg_time_jac_200 = compute_avg_time(csv_jac_200, max_iter)
    avg_time_mse = compute_avg_time(csv_mse, max_iter)
    # avg_time_rand = compute_avg_time(csv_rand, max_iter)

    print(f"Avg time to {max_iter} iters (s):")
    # print("JAC (1):", avg_time_jac_1 / 60)
    # print("JAC (10):", avg_time_jac_10 / 60)
    # print("JAC (50):", avg_time_jac_50 / 60)
    # print("JAC (100):", avg_time_jac_100 / 60)
    print("JAC (100):", avg_time_jac_200 / 60)
    print("MSE:", avg_time_mse/ 60)
    # print("Simulator:", avg_time_rand/ 60)



    iters = np.arange(1, max_iter + 1)

    # Define layout
    optimization_metrics = ["loss", "regularization"]
    inversion_metrics = ["inversion_MSE", "SSIM"]

    nrows, ncols = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 15))
    axes = axes.flatten()

    # Add row-level suptitles
    # fig.text(0.5, 0.955, "Least-squares optimization", ha='center', fontsize=16, weight='bold')
    # fig.text(0.5, 0.635, "Inversion", ha='center', fontsize=16, weight='bold')
    # fig.suptitle("Least-squares optimization", fontsize=18, weight='bold', y=0.965)

    # Use fig.text for other rows with adjusted vertical positions
    # fig.text(0.5, 0.625, "Inversion", ha='center', fontsize=16, weight='bold')

    # Inner function for standard metrics
    def plot_metric(ax, metric):
        print("metric", metric)
        # grouped_jac_1 = df_jac_1.groupby("iteration")[metric].agg(["mean"]).reset_index()[:max_iter]
        # grouped_jac_10 = df_jac_10.groupby("iteration")[metric].agg(["mean"]).reset_index()[:max_iter]
        # grouped_jac_50 = df_jac_50.groupby("iteration")[metric].agg(["mean"]).reset_index()[:max_iter]
        # grouped_jac_100 = df_jac_100.groupby("iteration")[metric].agg(["mean"]).reset_index()[:max_iter]
        grouped_jac_200 = df_jac_200.groupby("iteration")[metric].agg(["mean"]).reset_index()[:max_iter]
        grouped_mse = df_mse.groupby("iteration")[metric].agg(["mean"]).reset_index()[:max_iter]
        grouped_rand = df_rand.groupby("iteration")[metric].agg(["mean"]).reset_index()[:max_iter]

        # val_jac_1   = get_final_value(grouped_jac_1, "JAC (1)")
        # val_jac_10  = get_final_value(grouped_jac_10, "JAC (10)")
        # val_jac_50  = get_final_value(grouped_jac_50, "JAC (50)")
        # val_jac_100 = get_final_value(grouped_jac_100, "JAC (100)")
        val_jac_200 = get_final_value(grouped_jac_200, "JAC (200)")
        val_mse     = get_final_value(grouped_mse, "MSE")
        val_rand    = get_final_value(grouped_rand, "Simulator")

        color_jac, color_mse, color_rand = color_map.get(metric, ("blue", "orange", "gray"))
        # ax.plot(grouped_jac_1["iteration"], grouped_jac_1["mean"], label="Jvp:FIM (1)", color=color_jac, linestyle='-', marker='x', markevery=200)
        # ax.plot(grouped_jac_10["iteration"], grouped_jac_10["mean"], label="Jvp:FIM (10)", color=color_jac, linestyle='--', marker='D', markevery=200)
        # ax.plot(grouped_jac_50["iteration"], grouped_jac_50["mean"], label="Jvp:FIM (50)", color=color_jac, linestyle='-.', marker='o', markevery=200)
        # ax.plot(grouped_jac_100["iteration"], grouped_jac_100["mean"], label="Jvp:FIM (100)", color=color_jac, linestyle=':', marker='*', markevery=200)
        ax.plot(grouped_jac_200["iteration"], grouped_jac_200["mean"], label="Jvp:FIM (200)", color=color_jac, linestyle=':', marker='*', markevery=200)
        ax.plot(grouped_mse["iteration"], grouped_mse["mean"], label="MSE", color=color_mse, marker='s', markevery=200)
        ax.plot(grouped_rand["iteration"], grouped_rand["mean"], label="Simulator", color=color_rand, marker='^', markevery=200)
        ax.set_xlabel("iteration")
        ax.set_ylabel(index_name[metric])
        ax.set_title(metric_name[metric], fontsize=15)
        ax.grid(True)
        ax.legend(fontsize=12)

    def get_final_value(df_grouped, label):
        final_value = df_grouped[df_grouped["iteration"] == max_iter - 1]["mean"].values
        if len(final_value) > 0:
            print(f"{label} at iter={max_iter}: {final_value[0]:.4e}")
            return final_value[0]
        else:
            print(f"{label} at iter={max_iter}: not found")
            return None


    # Plot top (optimization)
    # for idx, metric in enumerate(optimization_metrics):
    #     plot_metric(axes[idx], metric)

    # Plot middle (inversion)
    for idx, metric in enumerate(inversion_metrics):
        plot_metric(axes[idx + 2], metric)

    # Plot bottom row: norm errors
    ax = axes[4]
    # ax.plot(iters, rel_2_1,   label="Jvp:FIM (1)", linestyle='-', marker='x', color="blueviolet", markevery=200)
    # ax.plot(iters, rel_2_10,  label="Jvp:FIM (10)", linestyle='--', marker='D', color="blueviolet", markevery=200)
    # ax.plot(iters, rel_2_50,  label="Jvp:FIM (50)", linestyle='-.', marker='o', color="blueviolet", markevery=200)
    # ax.plot(iters, rel_2_100, label="Jvp:FIM (100)", linestyle=':', marker='*', color="blueviolet", markevery=200)
    # ax.plot(iters, rel_2_mse, label="MSE", marker='s', color="red", markevery=200)
    # ax.errorbar(iters, rel_2_1_mean, yerr=rel_2_1_std, label="Jvp:FIM (1)", linestyle='-', marker='x', color="darkorange", errorevery=200, markevery=200,capsize=3,alpha=0.8)
    # ax.errorbar(iters, rel_2_10_mean, yerr=rel_2_10_std, label="Jvp:FIM (10)", linestyle='--', marker='D', color="forestgreen", errorevery=200, markevery=200,capsize=3,alpha=0.8)
    # ax.errorbar(iters, rel_2_50_mean, yerr=rel_2_50_std, label="Jvp:FIM (50)", linestyle='-.', marker='o', color="cornflowerblue", errorevery=200, markevery=200,capsize=3,alpha=0.8)
    # ax.errorbar(iters, rel_2_100_mean, yerr=rel_2_100_std, label="Jvp:FIM (100)", linestyle=':', marker='*', color="blueviolet", errorevery=200, markevery=200,capsize=3,alpha=0.8)
    ax.errorbar(iters, rel_2_200_mean, yerr=rel_2_200_std, label="Jvp:FIM (200)", linestyle=':', marker='*', color="blueviolet", errorevery=200, markevery=200,capsize=3,alpha=0.8)
    ax.errorbar(iters, rel_2_mse_mean, yerr=rel_2_mse_std, label="MSE", marker='s', color="darkred", errorevery=200, markevery=200,capsize=3,alpha=0.8)
    ax.set_title(r"Relative 2-norm error to $\mathbf{a}_\text{NS}$")
    ax.set_ylabel(r"$\frac{\|\mathbf{a}_{\text{model}} - \mathbf{a}_{\text{NS}}\|_2}{\|\mathbf{a}_{\text{NS}}\|_2}$")
    ax.set_xlabel("iteration")
    ax.grid(True)
    ax.legend(fontsize=12)

    ax = axes[5]
    # ax.plot(iters, rel_inf_1,   label="Jvp:FIM (1)", linestyle='-', marker='x', color="blueviolet", markevery=200)
    # ax.plot(iters, rel_inf_10,  label="Jvp:FIM (10)", linestyle='--', marker='D', color="blueviolet", markevery=200)
    # ax.plot(iters, rel_inf_50,  label="Jvp:FIM (50)", linestyle='-.', marker='o', color="blueviolet", markevery=200)
    # ax.plot(iters, rel_inf_100, label="Jvp:FIM (100)", linestyle=':', marker='*', color="blueviolet", markevery=200)
    # ax.plot(iters, rel_inf_mse, label="MSE", marker='s', color="red", markevery=200)
    # ax.errorbar(iters, rel_inf_1_mean, yerr=rel_inf_1_std, label="Jvp:FIM (1)", linestyle='-', marker='x', color="darkorange", errorevery=200, markevery=200,capsize=3,alpha=0.8)
    # ax.errorbar(iters, rel_inf_10_mean, yerr=rel_inf_10_std, label="Jvp:FIM (10)", linestyle='--', marker='D', color="forestgreen", errorevery=200, markevery=200,capsize=3,alpha=0.8)
    # ax.errorbar(iters, rel_inf_50_mean, yerr=rel_inf_50_std, label="Jvp:FIM (50)", linestyle='-.', marker='o', color="cornflowerblue", errorevery=200, markevery=200,capsize=3,alpha=0.8)
    # ax.errorbar(iters, rel_inf_100_mean, yerr=rel_inf_100_std, label="Jvp:FIM (100)", linestyle=':', marker='*', color="blueviolet", errorevery=200, markevery=200,capsize=3,alpha=0.8)
    ax.errorbar(iters, rel_inf_200_mean, yerr=rel_inf_200_std, label="Jvp:FIM (200)", linestyle=':', marker='*', color="blueviolet", errorevery=200, markevery=200,capsize=3,alpha=0.8)
    ax.errorbar(iters, rel_inf_mse_mean, yerr=rel_inf_mse_std, label="MSE", marker='s', color="darkred", errorevery=200, markevery=200,capsize=3,alpha=0.8)
    
    ax.set_title(r"Relative $\infty$-norm error to $\mathbf{a}_\text{NS}$")
    ax.set_ylabel(r"$\frac{\|\mathbf{a}_{\text{model}} - \mathbf{a}_{\text{NS}}\|_\infty}{\|\mathbf{a}_{\text{NS}}\|_\infty}$")
    ax.set_xlabel("iteration")
    ax.grid(True)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    plt.show()



# File paths
initial_guess = "smooth" #"smooth"
norm = "infty" #np.inf #2
max_iter = 1999 #1200 #2200
folder = "."
gd_type = "NGD" # None

print("folder exists", flush=True)

csv_jac_1 = f"{folder}/loss_statistics_multiple_samples_JAC_1_{initial_guess}.csv"
csv_jac_10 = f"{folder}/loss_statistics_multiple_samples_JAC_10_{initial_guess}.csv"
csv_jac_50 = f"{folder}/loss_statistics_multiple_samples_JAC_50_{initial_guess}.csv"
csv_jac_100 = f"{folder}/loss_statistics_multiple_samples_JAC_100_{initial_guess}.csv"
csv_jac_200 = f"{folder}/loss_statistics_multiple_samples_JAC_200_{initial_guess}.csv"
csv_mse = f"{folder}/loss_statistics_multiple_samples_MSE_{initial_guess}.csv"
csv_rand = f"{folder}/loss_statistics_multiple_samples_Devito_{initial_guess}.csv"

h5_jac_1 = f"{folder}/inversion_history_JAC_1_{initial_guess}_{gd_type}.h5"
h5_jac_10 = f"{folder}/inversion_history_JAC_10_{initial_guess}_{gd_type}.h5"
h5_jac_50 = f"{folder}/inversion_history_JAC_50_{initial_guess}_{gd_type}.h5"
h5_jac_100 = f"{folder}/inversion_history_JAC_100_{initial_guess}_{gd_type}.h5"
h5_jac_200 = f"{folder}/inversion_history_JAC_200_{initial_guess}_{gd_type}.h5"
h5_mse = f"{folder}/inversion_history_MSE_{initial_guess}_{gd_type}.h5"
h5_rand = f"{folder}/inversion_history_Devito_{initial_guess}_{gd_type}.h5"

print("h5 rand", flush=True)


# Metrics to plot
metrics = ["loss", "inversion_MSE", "regularization", "SSIM"]
# metric_name = ["Data Residual", r"Inversion Metric: $\frac{\|\mathbf{a}^\ast - \mathbf{a}\|_F}{\|\mathbf{a^\ast}\|_F}$", "Regularization Term (Laplacian)", "Inversion Metric: SSIM"]
# index_name = [r"$\frac{1}{n}\sum^n_{i=0}(\mathbf{y}_\text{true} - \mathbf{y}_\text{pred})^2$", r"$\frac{\|\mathbf{a}^\ast - \mathbf{a}\|_F}{\|\mathbf{a^\ast}\|_F}$", "Regularization Term (Laplacian)", "Inversion Metric: SSIM"]
index_name = {
    "loss": r"$\frac{1}{n}\sum^n_{i=0}(\mathbf{y}_\text{true} - \mathbf{y}_\text{pred})^2$",
    "inversion_MSE": r"$\frac{\|\mathbf{a}^\ast - \mathbf{a}\|_F}{\|\mathbf{a^\ast}\|_F}$",
    "regularization": "Regularization Term (Laplacian)",
    "SSIM": "SSIM"
}

metric_name = {
    "loss": "Data Residual",
    "inversion_MSE": r"Relative Frobenius error to $\mathbf{a}^\ast$",
    "regularization": "Regularization Term (Laplacian)",
    "SSIM": "SSIM"
}


# Color map for each metric (JAC, MSE, RAND)
color_map = {
    "loss": ("blueviolet", "darkred", "blue"), #("#1f77b4", "#aec7e8", "#7f7f7f"),
    "inversion_MSE": ("blueviolet", "darkred", "blue"), #("#ff7f0e", "#ffbb78", "#c7c7c7"),
    "regularization": ("blueviolet", "darkred", "blue"), # ("#2ca02c", "#98df8a", "#bcbd22"),
    "SSIM": ("blueviolet", "darkred", "blue") #("#d62728", "#ff9896", "#8c564b")
}

# Global tick size
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams["axes.labelsize"] = 17


file_name = f"metrics_comparison_over_iter_updated.png"

print("start", flush=True)

# Generate plot
plot_metrics_comparison(
    csv_jac_1, csv_jac_10, csv_jac_50, csv_jac_100,
    csv_mse, csv_rand,
    h5_jac_1, h5_jac_10, h5_jac_50, h5_jac_100, h5_rand, h5_mse,
    metrics, color_map,
    file_name , metric_name, index_name
)
