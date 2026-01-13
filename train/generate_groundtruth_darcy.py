"""
Exact replication of the Groundwater Flow inverse problem
from Beskos et al. (2017, J. Comp. Phys.) and Lan & Marzouk (2014–2017).

PDE:   -∇·(exp(u) ∇p) = 1    on [0,1]^2
BCs:   p(x,0)=x1,  p(x,1)=1-x1,  ∂p/∂x1|x1=0,1 = 0
Prior: Fourier–KL expansion of log-permeability u(x)
Obs.:  pressure at 36 points on circle r=0.5 + center, noise σ_y^2=1e-4
"""

import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from utils_inversion import *
from utils_plot import *
from groundwater.utils import GaussianRandomField, plot_fields

# ---------------------------------------------------------------------
# Generate dataset
# ---------------------------------------------------------------------
def generate_dataset(
    nx=128,
    ny=128,
    ncmp=8,
    s=1.,
    alpha=1.0,
    sigma=1.5,
    noise_var=1e-4,
    seed=1,
    device="cpu",
    out_path="ellipticPDE_Beskos.h5",
):

    # --- True coefficients (same as MATLAB: sin(i^2 + j^2))
    i, j = np.meshgrid(np.arange(1, ncmp + 1), np.arange(1, ncmp + 1), indexing="ij")
    # theta_truth = np.sin(i**2 + j**2).reshape(-1)
    # theta_truth = torch.tensor(theta_truth, dtype=torch.float32)

    theta_truth = torch.tensor(np.random.randn(ncmp**2).astype(np.float32))
    print("theta_truth", theta_truth.shape)

    # grf = GaussianRandomField(2, 128, alpha=1.1, tau=1e-6, sigma=1) #alpha = 1.1
    # theta_truth= grf.sample(1)
    theta_truth = torch.tensor(theta_truth)


    # --- Construct true field
    u_truth, k_truth = reconstruct_fourier_field(theta_truth, nx, ny, ncmp, s, alpha, sigma, device)
    print("u_truth", u_truth.shape)

    # --- Forward solve (your Devito groundwater model)
    forcing_term = torch.ones(nx, ny)
    gw_torch_model = GroundwaterModel(forcing_term.shape[0])
    def forward_with_tsteps(u, f, time_steps=50000):
        eq = gw_torch_model.groundwater_eq
        orig = eq.eval_fwd_op

        def wrapped(f_, u_, *args, **kwargs):
            kwargs.pop("time_steps", None)
            return orig(f_, u_, time_steps=time_steps, *args, **kwargs)

        eq.eval_fwd_op = wrapped
        try:
            return gw_torch_model(u, f)
        finally:
            eq.eval_fwd_op = orig
    y_truth = forward_with_tsteps(u_truth, forcing_term).cpu().numpy()

    # --- Observation geometry
    obs_pts = circular_observation_points()
    y_obs = sample_pressure_at_points(y_truth, obs_pts)
    y_noisy = y_obs + np.sqrt(noise_var) * np.random.randn(*y_obs.shape)

    # --- Save all results
    with h5py.File(out_path, "w") as f:
        f.create_dataset("theta_truth", data=theta_truth.numpy())
        f.create_dataset("u_truth", data=u_truth.cpu().numpy())
        f.create_dataset("k_truth", data=k_truth.cpu().numpy())
        f.create_dataset("y_field", data=y_truth)
        f.create_dataset("y_obs", data=y_noisy)
        f.create_dataset("obs_pts", data=obs_pts)
        f.attrs.update(dict(nx=nx, ny=ny, ncmp=ncmp, s=s, alpha=alpha, sigma=sigma,
                            noise_var=noise_var, seed=seed))

    print(f"✅ Dataset saved to {out_path}")


# ---------------------------------------------------------------------
# Example run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    seed = 9
    np.random.seed(seed)
    torch.manual_seed(seed)
    index = seed
    generate_dataset(out_path=f"ellipticPDE_Beskos_{index}.h5")

    # --------------------------------------------------
    # load the dataset
    # --------------------------------------------------
    fname = f"ellipticPDE_Beskos_{index}.h5"   # path to your saved file
    with h5py.File(fname, "r") as f:
        u_truth = f["u_truth"][:]
        k_truth = f["k_truth"][:]
        y_field = f["y_field"][:]
        obs_pts = f["obs_pts"][:]

    print("Loaded:", fname)
    print("  u_truth shape:", u_truth.shape)
    print("  y_field shape:", y_field.shape)
    print("  #obs =", len(obs_pts))

    # --------------------------------------------------
    # plotting utility
    # --------------------------------------------------
    def imshow_field(ax, field, title, cmap="viridis", cbar_label=None, vmin=None, vmax=None):
        im = ax.imshow(field.T, origin="lower", cmap=cmap,
                    extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label:
            cbar.set_label(cbar_label)
        return ax

    # --------------------------------------------------
    # make the figure (similar layout to Beskos Fig. 2)
    # --------------------------------------------------

    # plot_single(u_field, "beskos_groundwater_ufield", vmin=u_field.min(), cmap="RdBu_r")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    # log-permeability
    imshow_field(axs[0], u_truth, "log-permeability $u(x)$", cmap="RdBu_r",
                cbar_label="log(k)")

    # permeability with contours (cool colormap)
    ax = axs[1]
    im = ax.imshow(k_truth.T, origin="lower", cmap="cool",
                extent=[0, 1, 0, 1])
    # Add contour lines for structure
    contours = ax.contour(k_truth.T, levels=9, colors="k",
                        linewidths=0.5, alpha=0.7, extent=[0, 1, 0, 1])
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.2f", colors="k")

    ax.set_title("permeability $k(x)=e^{u(x)}$", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("k", rotation=0, labelpad=10)


    # pressure with observation points
    imshow_field(axs[2], y_field, "pressure field $p(x)$", cmap="coolwarm",
                cbar_label="pressure")
    # axs[2].scatter(obs_pts[:, 0], obs_pts[:, 1],
                # c="k", s=20, edgecolor="white", label="observations")
    # axs[2].legend(loc="upper right")

    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.suptitle("Beskos et al. (2017) Groundwater Flow Setup", fontsize=14)
    plt.savefig(f"beskos_groundwater_fields_{index}.png", dpi=300, bbox_inches="tight")



