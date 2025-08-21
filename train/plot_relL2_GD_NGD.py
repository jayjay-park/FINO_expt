import numpy as np
import h5py
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
initial_guess = "prior_mean"
folder_NGD = "prior_mean_wonoise_NGD"
folder_GD = "prior_mean_wonoise"
len_traj = 999

# --- PATHS ---
h5_jac_50 = f"{folder_GD}/inversion_history_JAC_50_{initial_guess}.h5"  # Jvp GD
h5_jac = f"{folder_GD}/inversion_history_JAC_400_{initial_guess}.h5"  # Jvp GD
h5_mse = f"{folder_GD}/inversion_history_MSE_{initial_guess}.h5"      # MSE GD
h5_devito = f"{folder_NGD}/inversion_history_Devito_{initial_guess}_NGD.h5"  # Numerical NGD

# --- LOAD TRAJECTORIES ---
with h5py.File(h5_jac_50, "r") as f:
    path_jvp_gd_50 = f["a"][:]
with h5py.File(h5_jac, "r") as f:
    path_jvp_gd = f["a"][:]
with h5py.File(h5_mse, "r") as f:
    path_mse_gd = f["a"][:]
with h5py.File(h5_devito, "r") as f:
    path_ngd = f["a"][:]

# --- FLATTEN ---
X_jvp_gd = path_jvp_gd.squeeze(0).reshape(path_jvp_gd.shape[1], -1)[:len_traj]
X_mse_gd = path_mse_gd.squeeze(0).reshape(path_mse_gd.shape[1], -1)[:len_traj]
X_ngd = path_ngd.squeeze(0).reshape(path_ngd.shape[1], -1)[:len_traj]

# --- RELATIVE L2 ERROR FUNCTION ---
def relative_l2_error(pred, ref):
    return np.linalg.norm(pred - ref, ord=2) / np.linalg.norm(ref, ord=2)

# --- COMPUTE ERRORS FOR ALL ITERATES ---
errors_jvp_vs_ngd = [relative_l2_error(X_jvp_gd[i], X_ngd[i]) for i in range(len(X_ngd))]
errors_mse_vs_ngd = [relative_l2_error(X_mse_gd[i], X_ngd[i]) for i in range(len(X_ngd))]

# --- PLOT ---
plt.figure(figsize=(8, 5))
plt.plot(errors_jvp_vs_ngd, label="Jvp (GD) vs Numerical NGD", color="blue")
plt.plot(errors_mse_vs_ngd, label="MSE (GD) vs Numerical NGD", color="red")
plt.xlabel("Iteration")
plt.ylabel("Relative L2 Error")
plt.title("Relative L2 Error Across Iterations")
plt.legend()
plt.grid(True)
plt.savefig("model_iter_ood.png")
