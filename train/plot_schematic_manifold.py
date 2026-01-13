#!/usr/bin/env python3
import os
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
import matplotlib.pyplot as plt

# =========================
# User-provided parameters
# =========================
initial_guess = "prior_mean"
type_opt = "GD"
folder = "prior_mean_noise(0.01)"   # where the .h5 files live
dest_folder = "prior_mean_noise(0.01)"
expt_name = "output_indist"
output = True   # True -> read dataset "u"; False -> read dataset "a"

# Optional knobs (safe defaults)
max_points_tsne = 20000        # subsample for t-SNE speed if needed
random_state = 42              # reproducibility across runs
dpi = 220                      # plot quality

# =========================
# Build file list (yours)
# =========================
if output is True:
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

# =========================
# HDF5 loading tailored to your layout
# =========================
def load_traj_flat(path: str, prefer_output: bool = True) -> np.ndarray:
    """
    Load a single HDF5 trajectory and flatten to [T, D].
    - If prefer_output=True, tries dataset 'u'; otherwise 'a'.
    - Accepts [T,H,W] or [H,W,T]; returns [T, H*W].
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        key = 'u' if prefer_output else 'a'
        if key not in f:
            # fallback: if the other exists, use it; else look for any 3D dataset
            other = 'a' if prefer_output else 'u'
            if other in f:
                key = other
            else:
                # find any 3D numeric dataset
                cand = None
                for k, ds in f.items():
                    if isinstance(ds, h5py.Dataset) and ds.ndim == 3 and np.issubdtype(ds.dtype, np.number):
                        cand = k; break
                if cand is None:
                    raise KeyError(f"No 3D numeric dataset found in {path}. Keys: {list(f.keys())}")
                key = cand

        arr = f[key][...]   # expect [T,H,W] or [H,W,T]
    arr = np.asarray(arr).squeeze()
    if arr.ndim == 2:
        arr = arr[None, ...]          # [1,H,W]
    elif arr.ndim != 3:
        raise ValueError(f"Unexpected shape {arr.shape} in {path} (expected [T,H,W] or [H,W,T])")

    # Make time the first axis
    H, W = arr.shape[-2], arr.shape[-1]
    # Heuristic: if last axis is the smallest, assume it's time dimension [H,W,T]
    if arr.shape[-1] < min(H, W):     # [H,W,T] -> [T,H,W]
        arr = np.moveaxis(arr, -1, 0)

    T = arr.shape[0]
    flat = arr.reshape(T, -1)         # [T, H*W]
    return flat

# =========================
# Collect & combine data
# =========================
rows = []
meta = []  # (model, file, dataset_used, n_samples, n_features)

for model, path in h5_files.items():
    if not os.path.exists(path):
        warnings.warn(f"Missing file, skipping: {path}")
        continue
    try:
        X = load_traj_flat(path, prefer_output=output)  # [T, D]
    except Exception as e:
        warnings.warn(f"Failed to read from {path}: {e}")
        continue

    n, d = X.shape
    meta.append((model, path, "u" if output else "a", n, d))
    df = pd.DataFrame(X)
    df["model"] = model
    rows.append(df)

if not rows:
    raise SystemExit("No data loaded. Check file paths and dataset contents ('u' or 'a').")

data = pd.concat(rows, ignore_index=True)
feature_cols = [c for c in data.columns if isinstance(c, (int, np.integer)) or str(c).isdigit()]
# Rename integer columns to f0, f1, ...
rename_map = {c: f"f{int(c)}" for c in feature_cols}
data.rename(columns=rename_map, inplace=True)
feature_cols = [rename_map[c] for c in feature_cols]

# =========================
# Preprocess
# =========================
X_all = data[feature_cols].values.astype(np.float32)
scaler = StandardScaler()
X_std = scaler.fit_transform(X_all)

# Sanity prints
models = sorted(data["model"].unique().tolist())
print("Per-model sample counts:")
for m in models:
    print(f"  {m}: {(data['model'].values == m).sum()} samples")
print(f"Total samples: {X_std.shape[0]} | Dim: {X_std.shape[1]}")

# Helper to pick stable params from N
def smart_k(N: int) -> int:
    # 5..min(50, N-1), scale ~ N^(1/3)
    return max(5, min(50, int(round(N ** (1/3)))))
def smart_perp(N: int) -> int:
    # 5..50 and strictly < N
    return max(5, min(50, max(5, (N - 1) // 3)))

N = X_std.shape[0]
k_iso = min(smart_k(N), N - 1) if N > 1 else 1
perp = smart_perp(N)

# Optional downsample for t-SNE speed
rng = np.random.default_rng(seed=random_state)
if X_std.shape[0] > max_points_tsne:
    idx_tsne = rng.choice(X_std.shape[0], size=max_points_tsne, replace=False)
else:
    idx_tsne = np.arange(X_std.shape[0])

# =========================
# Embeddings (guarded)
# =========================
embeddings = {}

# PCA (always safe)
pca = PCA(n_components=2, random_state=random_state)
embeddings["PCA"] = pca.fit_transform(X_std)

# Isomap (needs N > k)
if N > 6:
    try:
        iso = Isomap(n_neighbors=k_iso, n_components=2)
        embeddings["Isomap"] = iso.fit_transform(X_std)
        print(f"Isomap: n_neighbors={k_iso} (N={N})")
    except Exception as e:
        warnings.warn(f"Isomap failed: {e}")
else:
    print(f"Isomap skipped (N={N} too small).")

# t-SNE (needs N > ~10)
if N > 10:
    tsne = TSNE(
        n_components=2,
        perplexity=min(perp, idx_tsne.size - 1) if idx_tsne.size > 10 else 5,
        learning_rate="auto",
        init="pca",
        metric="euclidean",
        random_state=random_state,
        max_iter=1000,   # <-- use max_iter, not n_iter
        verbose=1 if N > 2000 else 0,
    )
    Y_tsne = np.empty((X_std.shape[0], 2), dtype=np.float32)
    Y_tsne[idx_tsne] = tsne.fit_transform(X_std[idx_tsne])
    if idx_tsne.size < X_std.shape[0]:
        Y_tsne_full = pca.transform(X_std)
        mask = np.ones(X_std.shape[0], dtype=bool)
        mask[idx_tsne] = False
        Y_tsne[mask] = Y_tsne_full[mask]
    embeddings["t-SNE"] = Y_tsne
    print(f"t-SNE: perplexity={min(perp, idx_tsne.size - 1)} (N={N}, used={idx_tsne.size})")
else:
    print(f"t-SNE skipped (N={N} too small).")

# UMAP (optional)
try:
    import umap  # pip install umap-learn
    if N > 6:
        um = umap.UMAP(n_components=2, n_neighbors=k_iso, random_state=random_state)
        embeddings["UMAP"] = um.fit_transform(X_std)
        print(f"UMAP: n_neighbors={k_iso} (N={N})")
    else:
        print(f"UMAP skipped (N={N} too small).")
except Exception as e:
    print(f"UMAP not available or failed: {e}")

# =========================
# Plotting
# =========================
outdir = Path(dest_folder) / expt_name
outdir.mkdir(parents=True, exist_ok=True)

palette = plt.get_cmap("tab10")
color_map = {m: palette(i % 10) for i, m in enumerate(models)}

def scatter_plot(ax, XY, title):
    for m in models:
        mask = (data["model"].values == m)
        ax.scatter(XY[mask, 0], XY[mask, 1], s=6, alpha=0.8, label=m, c=[color_map[m]])
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=2, frameon=True, fontsize=8, loc="best")

# One figure per method (all models together)
for method, XY in embeddings.items():
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter_plot(ax, XY, f"{method} — {expt_name}")
    fig.tight_layout()
    fig.savefig(outdir / f"{expt_name}_{method.replace(' ', '_')}.png", dpi=dpi)
    plt.close(fig)

# Small multiples: one panel per model for each method
for method, XY in embeddings.items():
    fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(4*len(models), 4), squeeze=False)
    for j, m in enumerate(models):
        ax = axes[0, j]
        mask = (data["model"].values == m)
        ax.scatter(XY[mask, 0], XY[mask, 1], s=6, alpha=0.85, c=[color_map[m]])
        ax.set_title(f"{method} — {m}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(outdir / f"{expt_name}_{method.replace(' ', '_')}_by_model.png", dpi=dpi)
    plt.close(fig)

# =========================
# Quick run log
# =========================
meta_df = pd.DataFrame(meta, columns=["model", "file", "dataset", "n_samples", "n_features"])
meta_df.to_csv(outdir / f"{expt_name}_ingest_summary.csv", index=False)
print(f"Saved plots and summary to: {outdir.resolve()}")
print(meta_df)
