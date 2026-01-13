import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# Path to one of your saved samples
file_path = "/net/slimdata/jayjaydata2/DeFINO_Richard/datasets/datasets/dataset_VMB/sample_00002.h5"

# --- Load data ---
with h5py.File(file_path, "r") as f:
    x = np.array(f["x"])   # shape (nx, nz, 2)
    y = np.array(f["y"])   # shape (nx, nz)
    print("x", x.shape)
    # optional attributes
    bundle_id = f.attrs.get("bundle_id", "N/A")
    background_tag = f.attrs.get("background_tag", "N/A")

# --- Extract fields ---
v_true = x[0, :, :]       # ground truth
v_back = x[1, :, :]       # background model
rtm_img = y               # RTM preconditioned image

print("after loading", v_true.shape, v_back.shape, rtm_img.shape)

# --- Plot ---
plt.figure(figsize=(24, 4))

plt.subplot(1, 3, 1)
plt.imshow(v_true, cmap="rainbow", aspect="auto")
plt.title("Ground Truth (x[0])")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(v_back, cmap="rainbow", aspect="auto")
plt.title("Background (x[1])")
plt.colorbar()

plt.subplot(1, 3, 3)

vmax = np.percentile(np.abs(rtm_img), 98)
im = plt.imshow(rtm_img, cmap="gray", vmin=-vmax, vmax=vmax, aspect="auto")
plt.title("RTM (y)")
plt.axis("off")
plt.colorbar()

plt.suptitle(f"GroundTruth {bundle_id} | Background: {background_tag}")
plt.tight_layout()
plt.savefig("sample_00001_plot.png", dpi=200)
plt.show()
