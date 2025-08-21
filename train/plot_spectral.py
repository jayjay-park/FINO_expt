import numpy as np, matplotlib.pyplot as plt

# def load_and_average(path, window=10):
#     data = np.load(path)
#     psd  = data["psd"]                       # (n_snapshots, n_bins)
#     k    = data["k"]                         # (n_bins,)

#     print("psd shape from np.load", psd.shape, "k shape", k.shape)

#     # --- ensure k is 1-D --------------------------------------------
#     if k.ndim == 2:
#         k = k[0]                  # take the first row (all rows identical)

#     # --- average over non-overlapping windows of length `window`
#     n     = psd.shape[0]
#     nb    = n // window                      # full blocks
#     psd_b = psd[:nb*window].reshape(nb, window, -1).mean(axis=1)  # (nb, n_bins)

#     return k, psd_b.mean(axis=0)             # grand mean of the blocks

import numpy as np

# def load_and_average_1d(path, window=10):
#     """
#     Load a .npz file that *might* have 'k' and 'psd' swapped
#     and/or transposed, then return   k  (n_bins,)   and
#     the window-averaged PSD   psd_avg  (n_bins,).

#     The file is expected to contain either:
#         psd : (n_snapshots, n_bins)   and   k : (n_bins,)
#     or the two arrays accidentally swapped.
#     """
#     d = np.load(path)
#     a, b = d["psd"], d["k"]

#     # --- identify which is the PSD matrix and which is the k-vector ----
#     # Case 1:  a is 2-D (PSD),  b is 1-D (k)      ← everything is fine
#     if a.ndim == 2 and b.ndim == 1:
#         psd_mat, k_vec = a, b

#     # Case 2:  b is 2-D (PSD) and a is 1-D (k)    ← keys are swapped
#     elif b.ndim == 2 and a.ndim == 1:
#         psd_mat, k_vec = b, a

#     # Anything else is ambiguous / malformed
#     else:
#         raise ValueError(
#             f"Unrecognised shape combo: psd{a.shape}, k{b.shape}."
#             "  Expected one 2-D and one 1-D array."
#         )

#     # --- sanity check --------------------------------------------------
#     if psd_mat.shape[1] != k_vec.shape[0]:
#         # Sometimes the PSD was written transposed (n_bins, n_snapshots)
#         if psd_mat.shape[0] == k_vec.shape[0]:
#             psd_mat = psd_mat.T                      # fix orientation
#         else:
#             raise ValueError("PSD and k sizes don’t line up.")

#     # --- block-average across snapshots -------------------------------
#     n_snap   = psd_mat.shape[0]
#     n_blocks = n_snap // window
#     psd_b    = psd_mat[: n_blocks * window]          \
#                    .reshape(n_blocks, window, -1)    \
#                    .mean(axis=1)                     # (n_blocks, n_bins)

#     psd_avg = psd_b.mean(axis=0)                     # grand mean
#     return k_vec, psd_avg

import numpy as np

def load_and_average_1d(path, window=10):
    """
    Robust loader for .npz files that contain
        psd : (n_snapshots, n_bins)
        k   : (n_snapshots, n_bins)   ← duplicated row-wise
    or any of the earlier variants.
    Returns
        k_vec   : (n_bins,)
        psd_avg : (n_bins,)   window-averaged over snapshots
    """
    d = np.load(path)
    a, b = d["psd"], d["k"]

    # ---------------------------------------------------------------
    #  Identify which array is k and which is PSD
    # ---------------------------------------------------------------
    if a.ndim == 2 and b.ndim == 1:          # normal case
        psd_mat, k_vec = a, b

    elif b.ndim == 2 and a.ndim == 1:        # keys swapped
        psd_mat, k_vec = b, a

    elif a.ndim == 2 and b.ndim == 2:        # both 2-D (your file)
        # Check if a’s rows are all identical *and* strictly increasing
        if np.allclose(a, a[0]) and np.all(np.diff(a[0]) >= 0):
            k_vec, psd_mat = a[0], b
        elif np.allclose(b, b[0]) and np.all(np.diff(b[0]) >= 0):
            k_vec, psd_mat = b[0], a
        else:
            raise ValueError("Can’t tell which array is k.")

    else:
        raise ValueError(f"Unsupported shapes: psd{a.shape}, k{b.shape}")

    # ---------------------------------------------------------------
    #  Windowed average
    # ---------------------------------------------------------------
    n_snap   = psd_mat.shape[0]
    n_blocks = n_snap // window
    psd_b    = psd_mat[: n_blocks * window]          \
                   .reshape(n_blocks, window, -1)    \
                   .mean(axis=1)                     # (n_blocks, n_bins)

    psd_avg = psd_b.mean(axis=0)
    return k_vec, psd_avg



# ------------------------------------------------------------------

type_opt = "GD"

styles = {
    r"PDE (Devito)":            dict(color="#4477AA", linestyle="-",  lw=2, alpha=0.8),
    r"MSE":        dict(color="#EE6677", linestyle="-",  lw=2, alpha=0.8),
    r"JVP-FIM $r\!=\!50$":   dict(color="#AA3377", linestyle="--", lw=1.8, alpha=0.8),
    r"JVP-FIM $r\!=\!200$":  dict(color="#AA3377", linestyle="-.", lw=1.8, alpha=0.8),
    r"JVP-FIM $r\!=\!400$":  dict(color="#AA3377", linestyle=":",  lw=1.8, alpha=0.8),
}

file_map = [
    (f"psd_Devito_prior_mean_{type_opt}.npz",  r"PDE (Devito)"),
    (f"psd_MSE_prior_mean_{type_opt}.npz",     r"MSE"),
    (f"psd_JAC_50_prior_mean_{type_opt}.npz",  r"JVP-FIM $r\!=\!50$"),
    (f"psd_JAC_200_prior_mean_{type_opt}.npz", r"JVP-FIM $r\!=\!200$"),
    (f"psd_JAC_400_prior_mean_{type_opt}.npz", r"JVP-FIM $r\!=\!400$"),
]

# for fname, label in file_map:
#     k, psd_avg = load_and_average(fname, window=10)
#     plt.semilogy(k, psd_avg, label=label, **styles[label])

# plt.xlabel(r"$|k|$  (Radial wavenumber)")
# plt.ylabel("Mean power")
# plt.legend(frameon=False, fontsize=10)
# plt.tight_layout()
# plt.savefig("update_power_spectrum_overlay.png", dpi=300)

# ---------------------------------------------------------------------
#  Plot 1-D power spectra that were produced with `psd_flattened`
# ---------------------------------------------------------------------
import matplotlib.pyplot as plt
import os

plt.figure(figsize=(12, 3.0))

for fname, label in file_map:
    if not os.path.exists(fname):
        print(f"[warning] {fname} not found – skipping")
        continue

    k, psd_avg = load_and_average_1d(fname, window=1)

    # -- drop the DC component (k = 0) if present ---------------------
    if k.size > 1:
        k_plot   = k[1:]
        psd_plot = psd_avg[1:]
    else:                       # degenerate case: only k = 0 was saved
        k_plot   = k
        psd_plot = psd_avg

    plt.loglog(k_plot, psd_plot, label=label, **styles[label])

plt.xlabel(r"$k$  (1-D wavenumber index)")
plt.ylabel(r"Mean power  $|F(k)|^2$")
plt.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig("power_spectrum_1d_overlay.png", dpi=300)
plt.close()



# def log_bin(k, psd, bins_per_dec=8):
#     edges = np.logspace(np.log10(k[1]), np.log10(k[-1]),
#                         num=int(np.log10(k[-1]/k[1]))*bins_per_dec+1)
#     idx   = np.searchsorted(k, edges)
#     k_bin = [k[i:j].mean()  for i,j in zip(idx[:-1], idx[1:])]
#     p_bin = [psd[i:j].mean() for i,j in zip(idx[:-1], idx[1:])]
#     return np.array(k_bin), np.array(p_bin)

# W = 3000                                   # row length
# plt.figure(figsize=(4.0, 3.0))

# for fname, label in file_map:
#     k, psd_avg = load_and_average_1d(fname, window=10)

#     # 1. ignore DC, 2. stay below Nyquist of a single row
#     mask = (k > 0) & (k <= W//2)
#     k_plot, p_plot = k[mask], psd_avg[mask]

#     # 3. log-bin
#     k_bin, p_bin = log_bin(k_plot, p_plot, bins_per_dec=6)

#     plt.loglog(k_bin, p_bin, label=label, **styles[label])
#     plt.fill_between(k_bin, p_bin-std, p_bin+std, alpha=0.2)

# # reference inertial-range slope (optional)
# ref = 5e-5 * k_bin**(-5/3)
# plt.loglog(k_bin, ref, 'k--', lw=1.0, label=r'$k^{-5/3}$')

# plt.xlabel(r"$k$  (cycles / pixel)")
# plt.ylabel(r"Mean power  $|F(k)|^2$")
# plt.xlim(k_bin.min(), k_bin.max())
# plt.legend(frameon=False, fontsize=9)
# plt.tight_layout()
# plt.savefig("power_spectrum_1d_overlay.png", dpi=300)



# import numpy as np, matplotlib.pyplot as plt, os

# # ***********************************
# # helper: log-bin the noisy spectrum
# # ***********************************
# def log_bin(k, psd, bins_per_dec=6):
#     """Return bin-centres and average PSD in log-spaced bins."""
#     edges = np.logspace(np.log10(k[0]), np.log10(k[-1]),
#                         num=int(np.log10(k[-1]/k[0]))*bins_per_dec + 1)
#     idx   = np.searchsorted(k, edges)
#     k_bin = np.array([k[i:j].mean()  for i, j in zip(idx[:-1], idx[1:])])
#     p_bin = np.array([psd[i:j].mean() for i, j in zip(idx[:-1], idx[1:])])
#     return k_bin, p_bin

# # ***********************************
# # load → clip → bin
# # ***********************************
# def prep_curve(path, label, W, window=10):
#     k, psd = load_and_average_1d(path, window=window)
#     # drop DC and artefacts beyond the row Nyquist
#     mask = (k > 0) & (k <= W//2)
#     k, psd = k[mask], psd[mask]
#     return log_bin(k, psd)

# # --------------------------------------------------------------------
# # plotting
# # --------------------------------------------------------------------
# W        = 9000          # grid width
# ref_file = f"psd_Devito_prior_mean_GD.npz"   # ground-truth spectrum

# k_ref, p_ref = prep_curve(ref_file, "reference", W)

# fig, (ax_psd, ax_err) = plt.subplots(2, 1, figsize=(4.3, 4.6),
#                                      sharex=True,
#                                      gridspec_kw=dict(height_ratios=[3, 1]))

# ax_psd.loglog(k_ref, p_ref, **styles["PDE (Devito)"],
#               label="PDE (Devito)")

# for fname, label in file_map[1:]:            # skip the first (reference)
#     if not os.path.exists(fname):
#         print(f"[skip] {fname} not found")
#         continue

#     k, p = prep_curve(fname, label, W)

#     # spectrum
#     ax_psd.loglog(k, p, **styles[label], label=label)

#     # relative error (decibels)
#     err_db = 10 * np.log10(p / p_ref)        # + = over-estimate
#     ax_err.semilogx(k, err_db, **styles[label])

# # cosmetics
# ax_psd.set_ylabel(r"Power  $|F(k)|^2$")
# ax_psd.legend(frameon=False, fontsize=8)
# ax_psd.set_xlim(k_ref.min(), k_ref.max())

# ax_err.set_xlabel(r"$k$  (1-D wavenumber index)")
# ax_err.set_ylabel(r"$\Delta$PSD  [dB]")
# ax_err.axhline(0, color="k", lw=0.7)
# ax_err.set_ylim(-20, 20)

# plt.tight_layout()
# plt.savefig("power_spectrum_1d_clean.png", dpi=300)
