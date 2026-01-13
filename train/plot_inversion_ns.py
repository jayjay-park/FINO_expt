#!/usr/bin/env python3
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────
# 1) CONFIG
# ─────────────────────────────────────────────────────────────
folder = "noise(0.01)_NS_smooth_partial(0.1)_lr=0.5_sample=10" #"."
dest_folder = folder
initial   = "smooth"
gd_type   = "_GD"
max_iter  = 2500
every     = 100
expt_name = "input_outdist"
data_type = "NS"
os.makedirs(dest_folder, exist_ok=True)

METHODS = {
    "MSE-FNO":           f"{folder}/inversion_history_MSE_{data_type}_{initial}{gd_type}.h5",
    "Numerical Simulator": f"{folder}/inversion_history_Devito_{data_type}_{initial}{gd_type}.h5",
    # "PINO":              f"{folder}/inversion_history_PINO_{initial}{gd_type}.h5",
    r"FINO ($r = 400$)": f"{folder}/inversion_history_JAC_{data_type}_400_{initial}{gd_type}.h5",
}

# seaborn HUSL palette (one unique color per method)
# pal = sns.color_palette("husl", len(METHODS))
# method_list = list(METHODS.keys())
# styles = {m: dict(color=pal[i], marker="o", linestyle="-") for i, m in enumerate(method_list)}
# unified palette + style (colors from HUSL, markers/linestyles from plot_inversion.py)
pal = sns.color_palette("husl", 4)
method_list = ["MSE-FNO", "Numerical Simulator", "PINO", r"FINO ($r = 400$)"]

symbols = {
    r"FINO ($r = 400$)": dict(marker="^", linestyle=":"),
    "MSE-FNO":           dict(marker="d", linestyle="--"),
    "Numerical Simulator": dict(marker="v", linestyle=":"),
    "PINO":              dict(marker="s", linestyle="-."), 
}

styles = {m: dict(color=pal[i], **symbols[m]) for i, m in enumerate(method_list)}


# ─────────────────────────────────────────────────────────────
# 2) H5 HELPERS
# ─────────────────────────────────────────────────────────────
def try_load(h5f, candidates):
    for k in candidates:
        if k in h5f:
            return h5f[k][:]
    return None

def load_flat(path, key_candidates):
    with h5py.File(path, "r") as f:
        arr = try_load(f, key_candidates)
        if arr is None:
            return None, None, None
        # arr shape: (S, T, H, W) or (S, T, D)
        if arr.ndim == 4:
            S, T, H, W = arr.shape
            return arr.reshape(S, T, H*W), H, W
        elif arr.ndim == 3:
            S, T, D = arr.shape
            return arr, int(np.sqrt(D)), int(np.sqrt(D))
        else:
            raise ValueError(f"Unexpected array shape {arr.shape} for keys {key_candidates}")

def ensure_T_use(arrs, max_iter):
    # arrs: list of arrays each shaped (S, T, D) or None
    Ts = [A.shape[1] for A in arrs if A is not None]
    return min([max_iter] + Ts) if Ts else max_iter

# ─────────────────────────────────────────────────────────────
# 3) LOAD ALL NEEDED DATASETS
# We try to read, for each method:
#   a      : iterate (parameters)
#   u      : forward output (if present)
# From NS file (or any file if available):
#   a_true : ground-truth parameters
#   u_true : observed/true data
# ─────────────────────────────────────────────────────────────
# dataset key candidates (most common variations)
A_KEYS      = ["a"]                         # iterates
U_KEYS      = ["u", "output"]
A_TRUE_KEYS = ["a_true", "a_gt", "true", "target"]
U_TRUE_KEYS = ["u_true", "y_true", "obs", "y"]

A_by_m, U_by_m = {}, {}
H = W = None

for m, path in METHODS.items():
    print("m", m)
    A_by_m[m], Hm, Wm = load_flat(path, A_KEYS)
    if Hm is not None: H, W = Hm, Wm
    U_by_m[m], _, _    = load_flat(path, U_KEYS)

# Try to find truths (prefer from NS file)
a_true_all = None
u_true_all = None
ref_path = METHODS.get("Numerical Simulator")
if ref_path and os.path.exists(ref_path):
    with h5py.File(ref_path, "r") as f:
        a_true_all = try_load(f, A_TRUE_KEYS)
        u_true_all = try_load(f, U_TRUE_KEYS)

# If truths not found in NS, scan other files
if a_true_all is None or u_true_all is None:
    for p in METHODS.values():
        with h5py.File(p, "r") as f:
            if a_true_all is None:
                a_true_all = try_load(f, A_TRUE_KEYS)
            if u_true_all is None:
                u_true_all = try_load(f, U_TRUE_KEYS)
            if a_true_all is not None and u_true_all is not None:
                break

# Flatten truths if present
def flatten_truth(truth):
    if truth is None: return None
    if truth.ndim == 4:
        S, T, Ht, Wt = truth.shape
        return truth.reshape(S, T, Ht*Wt)
    elif truth.ndim == 3:
        return truth
    elif truth.ndim == 2:
        # (S, D) -> tile across T later
        return truth[:, None, :]
    else:
        return None

a_true_all = flatten_truth(a_true_all)
u_true_all = flatten_truth(u_true_all)

# If no truths, we can still do relative errors vs NS
A_NS = A_by_m.get("Numerical Simulator", None)
U_NS = U_by_m.get("Numerical Simulator", None)

# unified T across what we’ll plot
T_use = ensure_T_use([A for A in A_by_m.values() if A is not None], max_iter)

# ─────────────────────────────────────────────────────────────
# 4) METRIC COMPUTATIONS (per-sample → mean±std across samples)
# ─────────────────────────────────────────────────────────────
def mse_mean_std(A, B):
    """Mean/std over samples of per-iteration MSE. Shapes (S, T, D)."""
    S = min(A.shape[0], B.shape[0])
    T = min(A.shape[1], B.shape[1], T_use)
    diff2 = (A[:S, :T] - B[:S, :T])**2
    mse_st = diff2.mean(axis=-1)     # (S, T)
    return mse_st.mean(axis=0), mse_st.std(axis=0)

def rel_p_mean_std(A, B, p=2):
    """Relative p-norm vs B, mean/std over samples. Shapes (S, T, D)."""
    S = min(A.shape[0], B.shape[0])
    T = min(A.shape[1], B.shape[1], T_use)
    if p == np.inf:
        num = np.max(np.abs(A[:S,:T] - B[:S,:T]), axis=-1)
        den = np.maximum(np.max(np.abs(B[:S,:T]), axis=-1), 1e-16)
    else:
        num = np.linalg.norm(A[:S,:T] - B[:S,:T], ord=p, axis=-1)
        den = np.maximum(np.linalg.norm(B[:S,:T], ord=p, axis=-1), 1e-16)
    rel = num / den                            # (S, T)
    return rel.mean(axis=0), rel.std(axis=0)

def grad2(field, h, w):
    f2d = field.reshape(h, w)
    gx = np.zeros_like(f2d); gy = np.zeros_like(f2d)
    gx[:,1:-1] = (f2d[:,2:] - f2d[:,:-2]) / 2.0
    gy[1:-1,:] = (f2d[2:,:] - f2d[:-2,:]) / 2.0
    return gx, gy

def H1_norm_flat(flat, h, w):
    f2d = flat.reshape(h, w)
    gx, gy = grad2(flat, h, w)
    return np.sqrt(np.sum(f2d**2) + np.sum(gx**2) + np.sum(gy**2))

def rel_H1_mean_std(A, B, h, w):
    S = min(A.shape[0], B.shape[0])
    T = min(A.shape[1], B.shape[1], T_use)
    rel = np.zeros((S, T))
    for s in range(S):
        for t in range(T):
            num = H1_norm_flat(A[s,t], h, w) - 0.0  # explicit
            num = H1_norm_flat(A[s,t] - B[s,t], h, w)
            den = max(H1_norm_flat(B[s,t], h, w), 1e-16)
            rel[s,t] = num / den
    return rel.mean(axis=0), rel.std(axis=0)

# Containers for plots (method -> (μ, σ))
DATA_RESIDUAL   = {}
MODEL_MSE       = {}
REL2_vs_NS      = {}
RELINF_vs_NS    = {}
RELH1_vs_NS     = {}

# 4a) Data residual: needs u_pred vs u_true.
#     If u_true missing, fall back to u vs U_NS (proxy).
for m, path in METHODS.items():
    A = A_by_m[m]
    U = U_by_m[m]
    if U is None:
        # cannot compute residual for this method
        continue

    # Preferred: u_true available
    if u_true_all is not None:
        mu, sd = mse_mean_std(U, u_true_all)
    elif U_NS is not None:
        # Fall back: compare to Numerical Simulator outputs (proxy for data)
        mu, sd = mse_mean_std(U, U_NS)
    else:
        # give up for this method
        continue
    DATA_RESIDUAL[m] = (mu, sd)

# 4b) Model error (MSE in parameter space): a_pred vs a_true if available.
#     If a_true missing, we do NOT fake it; we’ll skip ribbons for that method.
for m, path in METHODS.items():
    A = A_by_m[m]
    if A is None: continue
    if a_true_all is not None:
        mu, sd = mse_mean_std(A, a_true_all)
        MODEL_MSE[m] = (mu, sd)
    # else: skip (safer than an incorrect proxy)

# 4c) Relative errors vs NS in parameter space (sample+iter matched)
if A_NS is not None:
    for m, path in METHODS.items():
        A = A_by_m[m]
        if A is None: continue
        r2_mu, r2_sd       = rel_p_mean_std(A, A_NS, p=2)
        rinf_mu, rinf_sd   = rel_p_mean_std(A, A_NS, p=np.inf)
        REL2_vs_NS[m]      = (r2_mu, r2_sd)
        RELINF_vs_NS[m]    = (rinf_mu, rinf_sd)
        if H is not None and W is not None:
            rH1_mu, rH1_sd = rel_H1_mean_std(A, A_NS, H, W)
            RELH1_vs_NS[m] = (rH1_mu, rH1_sd)

iters = np.arange(min(T_use, max_iter))

# ─────────────────────────────────────────────────────────────
# 5) UNIFIED PLOTTING (mean ± std ribbons everywhere)
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size':       14,
    'lines.linewidth': 2,
    'lines.markersize': 10,
})

def plot_ribbon(panel_dict, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="-")

    for m in method_list:
        if m not in panel_dict:   # method missing for this metric
            continue
        mu, sd = panel_dict[m]
        mu, sd = mu[:len(iters)], sd[:len(iters)]
        st = styles[m]
        ax.plot(iters, mu, label=m, markevery=every, **st)
        ax.fill_between(iters, np.maximum(mu - sd, 1e-16), mu + sd,
                        alpha=0.3, color=st["color"])

    ax.set_title(title, fontweight="bold", fontsize=16)
    ax.set_xlabel("Iteration"); ax.set_ylabel(ylabel)
    ax.legend(fontsize=12)
    plt.tight_layout()
    out = f"{dest_folder}/{fname}_{expt_name}{gd_type}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)

# Panels (only plot if we have data)
# if DATA_RESIDUAL:
#     plot_ribbon(
#         DATA_RESIDUAL,
#         "Data residual",
#         r'$\frac{1}{n}\sum(\mathbf{y}_\text{true}-\mathbf{y}_\text{pred})^2$',
#         "mt_loss_ribbon"
#     )
# if MODEL_MSE:
#     plot_ribbon(
#         MODEL_MSE,
#         "Model error in MSE",
#         r'$\frac{1}{n}\sum(\mathbf{a}_\text{true}-\mathbf{a}_\text{pred})^2$',
#         "mt_model_error_ribbon"
#     )
if REL2_vs_NS:
    plot_ribbon(
        REL2_vs_NS,
        r"Model Error in Relative $L_2$ (w.r.t. NS)",
        r'$\frac{\|\mathbf{a}_{nn}-\mathbf{a}_{ns}\|_2}{\|\mathbf{a}_{ns}\|_2}$',
        "mt_rel2_vsNS"
    )
if RELINF_vs_NS:
    plot_ribbon(
        RELINF_vs_NS,
        r"Model Error in Relative $L_\infty$ (w.r.t. NS)",
        r'$\frac{\|\mathbf{a}_{nn}-\mathbf{a}_{ns}\|_\infty}{\|\mathbf{a}_{ns}\|_\infty}$',
        "mt_relinf_vsNS"
    )
if RELH1_vs_NS:
    plot_ribbon(
        RELH1_vs_NS,
        r"Model Error in Relative $H^1$ (w.r.t. NS)",
        r'$\frac{\|\mathbf{a}_{nn}-\mathbf{a}_{ns}\|_{H^1}}{\|\mathbf{a}_{ns}\|_{H^1}}$',
        "mt_relH1_vsNS"
    )
