import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# Path to your file
h5_path = "datasets/dataset_DARCY_cov3/samples/sample_0.h5"

# Open and inspect the file
with h5py.File(h5_path, "r") as f:
    print("Available datasets:", list(f.keys()))
    
    # Load fields
    x   = f["x"][:]
    y   = f["y"][:]
    v   = f["v"][:]
    s   = f["s"][:]
    Jvp = f["Jvp"][:]
    L   = f["L"][:]

# Basic visualization of fields
def plot_field(field, title, filename=None):
    plt.figure(figsize=(6,6))
    im = plt.imshow(field, cmap="jet", origin="upper")
    plt.title(title)
    plt.colorbar(im, fraction=0.046)
    plt.axis("off")
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

# Plot input x and output y (assumed square fields)
plot_field(x, "Input x", "input_x.png")
plot_field(y, "Output y", "output_y.png")

# Optionally: plot first few Jvp modes
for i in range(min(3, Jvp.shape[1])):
    jvp_i = Jvp[:, i].reshape(x.shape)
    plot_field(jvp_i, f"Jvp Mode {i}", f"jvp_mode_{i}.png")

# Optionally: plot leading eigenvector v
for i in range(min(3, v.shape[1])):
    v_i = v[:, i].reshape(x.shape)
    plot_field(v_i, f"v Eigenvector {i}", f"v_mode_{i}.png")
