import numpy as np
import matplotlib.pyplot as plt

def sample_log_perm(nx=128, ny=128, ncmp=10, s=1.1, alpha=1.0, sigma=1.5, seed=None):
    """Draw one KL–Fourier sample of log-permeability."""
    if seed is not None:
        np.random.seed(seed)
    theta = np.random.randn(ncmp, ncmp)

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    u = np.zeros_like(X)
    for i in range(1, ncmp + 1):
        for j in range(1, ncmp + 1):
            lam = sigma * (alpha + np.pi**2 * (i**2 + j**2)) ** (-s / 2)
            phi = np.cos(np.pi * (i - 0.5) * X) * np.cos(np.pi * (j - 0.5) * Y)
            u += lam * theta[i - 1, j - 1] * phi
    return u

# -------------------------------------------------------------
# Generate N samples
# -------------------------------------------------------------
N = 6   # number of realizations
nx = ny = 128
ncmp = 8
s = 1.
alpha = 1.0
sigma = 1.5

samples = [sample_log_perm(nx, ny, ncmp, s, alpha, sigma, seed=48+i) for i in range(N)]

# Plot a few
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.ravel()
for k, ax in enumerate(axs):
    u = samples[k]
    im = ax.imshow(u.T, origin="lower", cmap="RdBu_r", extent=[0, 1, 0, 1])
    ax.set_title(f"log-perm sample {k+1}")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(f"{N} KL-truncated log-permeability samples (n_modes={ncmp})", fontsize=13)
plt.tight_layout()
plt.savefig("beskos_multi1")
