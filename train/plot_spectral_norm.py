import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import matplotlib.gridspec as gridspec

# ---------------------------
# CONFIGURATION
# ---------------------------
initial_guess = "prior_mean"
type_opt = "GD"
folder = "."
dest_folder = "prior_mean_noise(0.01)_ood(+5)"
expt_name = "output_outdist"
output = True
n_grid = 50
len_traj = 1000
num_bins = 100
log_eps = 1e-8
np.random.seed(42)

# HDF5 files and model labels
if output == True:
    h5_files = {
        "JVP (rank=50)":  f"{folder}/inversion_history_output_JAC_50_{initial_guess}_{type_opt}.h5",
        "JVP (rank=200)": f"{folder}/inversion_history_output_JAC_200_{initial_guess}_{type_opt}.h5",
        "JVP (rank=400)": f"{folder}/inversion_history_output_JAC_400_{initial_guess}_{type_opt}.h5",
        "MSE":            f"{folder}/inversion_history_output_MSE_{initial_guess}_{type_opt}.h5",
        "Devito":         f"{folder}/inversion_history_output_Devito_{initial_guess}_{type_opt}.h5",
    }
else:
    h5_files = {
        "JVP (rank=50)":  f"{folder}/inversion_history_JAC_50_{initial_guess}_{type_opt}.h5",
        "JVP (rank=200)": f"{folder}/inversion_history_JAC_200_{initial_guess}_{type_opt}.h5",
        "JVP (rank=400)": f"{folder}/inversion_history_JAC_400_{initial_guess}_{type_opt}.h5",
        "MSE":            f"{folder}/inversion_history_MSE_{initial_guess}_{type_opt}.h5",
        "Devito":         f"{folder}/inversion_history_Devito_{initial_guess}_{type_opt}.h5",
    }

# ---------------------------
# UTILITIES
# ---------------------------

def compute_radial_psd(field_2d, num_bins=100):
    F = np.fft.fftshift(np.fft.fft2(field_2d))
    power = np.abs(F) ** 2

    nx, ny = 128, 128
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2)

    k_bins = np.linspace(0, k_mag.max(), num_bins + 1)
    psd_avg = np.zeros(num_bins)

    for i in range(num_bins):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if np.any(mask):
            psd_avg[i] = power[mask].mean()
        else:
            psd_avg[i] = 0.0

    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    return k_centers, psd_avg

def load_trajectory(h5_path):
    with h5py.File(h5_path, "r") as f:
        if output == True:
            a_traj = f["u"][:].squeeze()  # shape: [T, H, W]
        else:
            a_traj = f["a"][:].squeeze()  # shape: [T, H, W]
        return a_traj

def compute_spectrum_evolution(a_traj, num_bins=100):
    return np.stack([
        compute_radial_psd(a_t.squeeze(), num_bins)[1]
        for a_t in a_traj
    ])

def compute_spectrum_evolution_over_time(traj, num_bins):
    """Compute log-normalized radial PSD over time. Returns [T, num_bins]"""
    T = traj.shape[0]
    spectra = []
    for t in range(T):
        field = traj[t]
        _, psd = compute_radial_psd(field, num_bins)
        psd_sum = np.sum(psd)
        psd_norm = psd / psd_sum if psd_sum > 0 else np.ones_like(psd) * log_eps
        psd_log = np.log10(psd_norm + log_eps)
        spectra.append(psd_log)
    return np.stack(spectra)  # [T, num_bins]

# ----------
# Plot
# ----------

def plot_final_spectrum_curves(spectra_dict, num_bins=100):
    plt.figure(figsize=(7, 5))

    for label, spectra in spectra_dict.items():
        final_spectrum = spectra[-1]  # last iteration
        k_vals = np.arange(num_bins)
        plt.plot(k_vals, final_spectrum + log_eps, label=label)

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Wavenumber bin")
    plt.ylabel("Power Spectrum (log scale)")
    if output == True:
        plt.title(r"Spectrum Comparison of $\mathbf{u_{1000}}$")
    else:
        plt.title(r"Spectrum Comparison of $\mathbf{a_{1000}}$")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()

    fname = f"{dest_folder}/final_spectrum_{expt_name}_{len_traj}_{type_opt}.png"
    plt.savefig(fname, dpi=200)
    print(f"Saved to: {fname}")
    plt.show()

def plot_all_spectra_evolution(spectra_dict, num_bins, len_traj, type_opt, expt_name):
    n_models = len(spectra_dict)
    fig = plt.figure(figsize=(4.5 * n_models, 4.5))
    gs = gridspec.GridSpec(1, n_models + 1, width_ratios=[1] * n_models + [0.05], wspace=0.2)

    vmin = min(s.min() for s in spectra_dict.values())
    vmax = max(s.max() for s in spectra_dict.values())

    axes = []
    for i, (label, spectra) in enumerate(spectra_dict.items()):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(spectra, aspect="auto", origin="lower", cmap="magma",
                       vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("Wavenumber bin")
        if i == 0:
            ax.set_ylabel("Iteration for Inversion")

        # Optional: Show x-axis as numeric values
        ax.set_xticks(np.linspace(0, num_bins - 1, 5))
        ax.set_xticklabels([f"{int(x)}" for x in np.linspace(0, num_bins - 1, 5)])

        axes.append(ax)

    # Add shared colorbar on far right
    cbar_ax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("log$_{10}$ normalized spectrum")

    plt.tight_layout()
    plt.savefig(f'{dest_folder}/spectral_evolution_normalized_{expt_name}_{len_traj}_{type_opt}.png', dpi=200)
    plt.show()


# ---------------------------
# MAIN
# ---------------------------

spectra_by_model = {}

for label, path in h5_files.items():
    if not Path(path).exists():
        print(f"[Warning] Missing file: {path}")
        continue

    print(f"Processing: {label}")
    traj = load_trajectory(path)  # shape [T, H, W]
    spectra = compute_spectrum_evolution(traj, num_bins)
    spectra_by_model[label] = spectra

plot_final_spectrum_curves(spectra_by_model, num_bins=num_bins)

spectra_by_model_over_iteration = {}

for label, path in h5_files.items():
    if not Path(path).exists():
        print(f"[Warning] Missing file: {path}")
        continue

    print(f"Processing: {label}")
    traj = load_trajectory(path)  # shape [T, H, W]
    spectra = compute_spectrum_evolution_over_time(traj, num_bins)
    spectra_by_model_over_iteration[label] = spectra

plot_all_spectra_evolution(spectra_by_model_over_iteration, num_bins, len_traj, type_opt, expt_name)

