import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def load_a(path, parameter='g'):
    with h5py.File(path,'r') as f:
        A = f[parameter][:]            # (samples, iters, H, W)
    s, t, h, w = A.shape
    return A.reshape(s, t, h*w)


# ----------------------
# Plotting Functions
# ----------------------

# def plot_single(true1, path, cmap="viridis", vmin=None, vmax=None, show_cbar=True):
#     plt.figure(figsize=(32, 10))
#     plt.rcParams.update({'font.size': 16})
#     # print("vmin", vmin, vmax)
#     if vmin == None:
#         norm = mcolors.CenteredNorm()
#     # else:
#     #     norm = colors.Normalize(vmin=vmin, vmax=vmax) if (vmin is not None and vmax is not None) else colors.CenteredNorm()
#     ny, nx = true1.shape
#     dx = dz = 12.5
#     extent = [0, nx*dx, ny*dz, 0]
    
#     fig, ax = plt.subplots()
#     if vmin == None:
#         cax = ax.imshow(true1, cmap=cmap, norm=norm, extent=extent, aspect="auto")
#     else:
#         cax = ax.imshow(true1, cmap=cmap, extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
#     if show_cbar == True:
#         # plt.colorbar(cax, ax=ax, fraction=0.045, pad=0.06)
#         plt.colorbar(cax, ax=ax, fraction=0.04, pad=0.06)
#     # ax.set_xticks([])
#     # ax.set_yticks([])
#     plt.xlabel("X (m)")
#     plt.ylabel("Depth (m)")
#     plt.savefig(path, dpi=150, bbox_inches='tight')
#     plt.close()

def plot_single(true1, path, cmap="viridis", vmin=None, vmax=None, show_cbar=True):
    plt.rcParams.update({'font.size': 16})
    
    ny, nx = true1.shape
    dx = dz = 12.5
    extent = [0, nx * dx, ny * dz, 0]

    # Create your actual figure with the correct shape
    fig, ax = plt.subplots(figsize=(18, 8))

    if vmin is None:
        norm = mcolors.CenteredNorm()
        cax = ax.imshow(true1, cmap=cmap, norm=norm, extent=extent, aspect="auto")
    else:
        cax = ax.imshow(true1, cmap=cmap, vmin=vmin, vmax=vmax,
                        extent=extent, aspect="auto")

    if show_cbar:
        fig.colorbar(cax, ax=ax, fraction=0.04, pad=0.06)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Depth (m)")
    
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_single_contour(true1, path, cmap="cool", vmin=None, vmax=None):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot image with consistent extent
    im = ax.imshow(true1, cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, 1, 0, 1], origin="lower")

    # Create contour lines with matching coordinate extent
    ny, nx = true1.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    contours = ax.contour(X, Y, true1, levels=3, colors="k", linewidths=0.8, alpha=0.7)
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.2f", colors="k")

    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.06)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_fields_triplet(U_t, V_t, idx, dim, name, dest_folder):
    """Plot simulator field, surrogate field, and misfit side by side."""
    sim = U_t[0, idx, :].reshape(dim, dim)
    sur = V_t[0, idx, :].reshape(dim, dim)
    mis = sim - sur

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    imgs = []
    vmax = np.max(np.abs(sim))
    vmin = np.min(np.abs(sim))
    imgs.append(axes[0].imshow(sim, cmap="viridis", vmin=-vmax, vmax=vmax))
    vmax = np.max(np.abs(sur))
    vmin = np.min(np.abs(sur))
    imgs.append(axes[1].imshow(sur, cmap="viridis", vmin=-vmax, vmax=vmax))
    vmax = np.max(np.abs(mis))
    imgs.append(axes[2].imshow(mis, cmap="RdBu_r", vmin=-vmax, vmax=vmax))

    titles = ["Simulator", "Surrogate", "Misfit (Sim - Sur)"]
    for ax, title, img in zip(axes, titles, imgs):
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"{dest_folder}/grad_triplet_{idx}_{name}.png", dpi=300)
    plt.close()

def plot_triplet(U_t, V_t, W_t, idx, dim, name, dest_folder):

    U_t = U_t.reshape(dim, dim).detach().cpu().numpy()
    V_t = V_t.reshape(dim, dim).detach().cpu().numpy()
    W_t = W_t.reshape(dim, dim).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    imgs = []
    vmax = np.max(np.abs(U_t))
    imgs.append(axes[0].imshow(U_t, cmap="viridis", vmin=-vmax, vmax=vmax))
    vmax = np.max(np.abs(V_t))
    imgs.append(axes[1].imshow(V_t, cmap="viridis", vmin=-vmax, vmax=vmax))
    vmax = np.max(np.abs(W_t))
    imgs.append(axes[2].imshow(W_t, cmap="viridis", vmin=-vmax, vmax=vmax))

    titles = [r"$g_{sim}$", r"$g_{sim}^{nat}$", r"$g_{nn}$"]
    for ax, title, img in zip(axes, titles, imgs):
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"{dest_folder}/grad_triplet_{idx}_{name}.png", dpi=300)
    plt.close()

def plot_observed_only_with_scatter(data, x_idx, y_idx, ax, cmap='jet'):
    # extract the value at each observation
    vals = data[y_idx, x_idx]

    # choose same norm logic
    vmin, vmax = vals.min(), vals.max()
    if vmin < 0 < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    sc = ax.scatter(x_idx, y_idx, c=vals, cmap=cmap, norm=norm, s=50, marker='s')
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    return sc


def plot_inversion_result(x0, x, true_y, y, x_pred, loss_type, index, x_idx, y_idx, iter, folder, data_type, sub_sampling=None, top_subsampling=False):
    # pull everything off‐GPU, to numpy:
    fields = [
        x0.detach().squeeze().cpu().numpy(),       # initial guess
        true_y.squeeze().cpu().numpy(),            # ground truth output
        x.squeeze().cpu().numpy(),                 # ground truth input
        y.squeeze().cpu().numpy(),                 # forward prediction
        x_pred,    # inversion result
        x.squeeze().cpu().numpy() - x_pred
    ]
    true = fields[1]
    x_range = fields[2]
    titles = [
        r'Initial Guess ($a_0$)',
        r'Observation ($y$)',
        r'Ground Truth Input ($a^\ast$)',
        r'Forward Prediction ($\hat{u}$)',
        r'Inversion Result ($a$)',
        r'$a - a^\ast$'
    ]

    # your observed locations
    x_idx = x_idx.detach().cpu().numpy()
    y_idx = y_idx.detach().cpu().numpy()

    fig, axes = plt.subplots(3, 2, figsize=(10,15))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        data = fields[i].squeeze()
        # norm for colorbar
        normal_norm = mcolors.Normalize(vmin=true.min(), vmax=true.max())
        normal_norm_x = mcolors.Normalize(vmin=x_range.min(), vmax=x_range.max())
        norm = mcolors.CenteredNorm()
        if data_type == "NS":
            cmap = "RdYlBu"
        else:
            cmap = 'RdYlBu'

        if i in (1, 3):  # only observed points
            if sub_sampling == True:
                sc = ax.scatter(x_idx, y_idx, c=data[y_idx, x_idx], cmap=cmap, norm=normal_norm, s=5, marker='o')
            else:
                sc = ax.scatter(x_idx, y_idx, c=data[y_idx, x_idx], cmap=cmap, norm=normal_norm, s=5, marker='o')
            mappable = sc
            # set the axes limits to match the image‐grid
            ax.set_xlim(-0.5, data.shape[1]-0.5)
            ax.set_ylim(data.shape[0]-0.5, -0.5)      # flip y so origin matches imshow
            ax.set_aspect('equal')

        else:  # full‐field image
            if data_type == "NS":
                im = ax.imshow(
                    data, cmap='BrBG' if i<5 else 'RdGy',
                    norm= norm, origin='lower',
                    extent=(0, data.shape[1], 0, data.shape[0]), aspect='equal'
                )
            else:
                im = ax.imshow(
                    data, cmap='BrBG' if i<5 else 'RdGy',
                    norm= normal_norm_x if i<5 else norm, origin='lower',
                    extent=(0, data.shape[1], 0, data.shape[0]), aspect='equal'
                )
            mappable = im

        ax.set_title(titles[i])
        # exactly one colorbar:
        fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if loss_type == "JAC" and top_subsampling == False:
        plt.savefig(f"{folder}/inversion_result_{loss_type}_{index}_{iter}.png")
    elif loss_type == "JAC" and top_subsampling == True:
        plt.savefig(f"{folder}_top/inversion_result_{loss_type}_{index}_{iter}.png")
    elif loss_type != "JAC" and top_subsampling == False:
        plt.savefig(f"{folder}/inversion_result_{loss_type}_{index}_{iter}.png")
    else:
        plt.savefig(f"{folder}_top/inversion_result_{loss_type}_{index}_{iter}.png")
    plt.close(fig)

