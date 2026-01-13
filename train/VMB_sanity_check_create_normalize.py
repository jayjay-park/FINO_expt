import os
import h5py
import numpy as np

# ============================================================
# CONFIG
# ============================================================
data_dir = "/net/slimdata/jayjaydata2/DeFINO_Richard/datasets/datasets/dataset_VMB_easy"
extend_rows = 13

# ============================================================
# HELPER FUNCTION
# ============================================================
def extend_constant_region(y, extend_rows=10, tol=1e-6):
    """
    Detect the number of initial rows that are constant,
    extend that region downward by `extend_rows`,
    and fill with the same constant value.
    """
    diffs = np.abs(np.diff(y, axis=0)).max(axis=1)
    const_rows = np.argmax(diffs > tol) if np.any(diffs > tol) else len(diffs)
    const_rows = min(const_rows + extend_rows, y.shape[0])
    y_mod = y.copy()
    y_mod[:const_rows, :] = y_mod[0, :].copy()
    return y_mod

# ============================================================
# GATHER FILES
# ============================================================
train_paths = sorted([
    os.path.join(data_dir, f)
    for f in os.listdir(data_dir)
    if f.endswith(".h5")
])
print(f"📁 Found {len(train_paths)} .h5 files in {data_dir}")

# ============================================================
# PROCESS FILES
# ============================================================
v_max = 0.0
y_min, y_max = float("inf"), float("-inf")

for i, p in enumerate(train_paths):
    if not os.path.isfile(p):
        continue

    with h5py.File(p, "r+") as f:
        # Load arrays
        x = f["x"][:].astype(np.float32)
        y = f["y"][:].astype(np.float32)

        # Create extended version
        y_ext = extend_constant_region(y, extend_rows=extend_rows)

        # Remove existing y_ext_13 (if re-running)
        if "y_ext_13" in f:
            del f["y_ext_13"]

        # Add new dataset
        f.create_dataset("y_ext_13", data=y_ext, compression="gzip")

        # Update global stats
        v_max = max(v_max, np.max(x[..., 0]), np.max(x[..., 1]))
        y_min = min(y_min, np.min(y_ext))
        y_max = max(y_max, np.max(y_ext))

    if (i + 1) % 50 == 0 or i == len(train_paths) - 1:
        print(f"  ✅ Processed {i + 1}/{len(train_paths)} files")

# ============================================================
# SAVE GLOBAL STATS
# ============================================================
np.savez("vmb_global_max_easy.npz", v_max=v_max, y_min=y_min, y_max=y_max)
print("\n✅ All files updated with `y_ext_13` added.")
print(f"✅ Velocity max = {v_max:.6f}")
print(f"✅ RTM min = {y_min:.6f}, RTM max = {y_max:.6f}")

# import os
# import h5py
# import numpy as np

# data_dir = "/net/slimdata/jayjaydata2/DeFINO_Richard/datasets/datasets/dataset_VMB"

# train_paths = sorted([
#     os.path.join(data_dir, f)
#     for f in os.listdir(data_dir)
#     if f.endswith(".h5")
# ])

# print(f"Found {len(train_paths)} .h5 files in {data_dir}")

# def extend_constant_region(y, extend_rows=10, tol=1e-6):
#     """
#     Detect the number of initial rows that are constant, extend that
#     region downward by `extend_rows`, and fill with the same constant value.
#     """
#     # Compute row-wise difference norm
#     diffs = np.abs(np.diff(y, axis=0)).max(axis=1)
#     const_rows = np.argmax(diffs > tol) if np.any(diffs > tol) else len(diffs)

#     # Constant region ends at const_rows (exclusive)
#     const_rows = min(const_rows + extend_rows, y.shape[0])
#     const_val = y[0, :].copy()
#     y[:const_rows, :] = const_val
#     return y

# def compute_global_max(sample_paths):
#     v_max = 0.0
#     y_max = 0.0
#     y_min = 0.0

#     for p in sample_paths:
#         if not os.path.isfile(p):
#             continue

#         with h5py.File(p, "r") as f:
#             x = f["x"][:].astype(np.float32)  # (nx, ny, 2)
#             y = f["y"][:].astype(np.float32)  # (nx, ny)
#             y_before = y.copy()

#         # --- handle constant region ---
#         y = extend_constant_region(y, extend_rows=13)

#         # Velocity
#         v_max = max(v_max, np.max(x[0, :, :]), np.max(x[1, :, :]))

#         # RTM min/max
#         y_max = max(y_max, np.max(y))
#         y_min = min(y_min, np.min(y))

#     return v_max, y_min, y_max, y, y_before

# v_max, y_min, y_max, y, y_before = compute_global_max(train_paths)
# np.savez("vmb_global_max.npz", v_max=v_max, y_min=y_min, y_max=y_max)

# import matplotlib.pyplot as plt
# plt.imshow(y, cmap="gray"); plt.colorbar(); plt.savefig("extended_row_rtm"); plt.close()
# plt.imshow(y_before, cmap="gray"); plt.colorbar(); plt.savefig("extended_row_rtm_before"); plt.close()


# print(f"✅ Velocity max = {v_max:.6f}")
# print(f"✅ RTM min = {y_min:.6f}, RTM max = {y_max:.6f}")



# import os
# import h5py
# import numpy as np

# # Path to your dataset folder
# data_dir = "/net/slimdata/jayjaydata2/DeFINO_Richard/datasets/datasets/dataset_VMB"

# # Gather all .h5 files
# train_paths = sorted([
#     os.path.join(data_dir, f)
#     for f in os.listdir(data_dir)
#     if f.endswith(".h5")
# ])

# print(f"Found {len(train_paths)} .h5 files in {data_dir}")

# def compute_global_max(sample_paths):
#     v_max = 0.0
#     y_max = 0.0

#     for p in sample_paths:
#         # Skip directories
#         if not os.path.isfile(p):
#             continue

#         with h5py.File(p, "r") as f:
#             x = f["x"][:].astype(np.float32)   # (nx, ny, 2)
#             y = f["y"][:].astype(np.float32)   # (nx, ny)

#         # Velocity: positive -> divide by its overall max
#         v_max = max(v_max, np.max(x[..., 0]), np.max(x[..., 1]))

#         # RTM: signed -> use max(|max|, |min|)
#         y_abs_max = np.abs(y).max()
#         y_max = max(y_max, y_abs_max)

#     return v_max, y_max

# # Compute and save
# v_max, y_max = compute_global_max(train_paths)
# np.savez("vmb_global_max.npz", v_max=v_max, y_max=y_max)
# print(f"✅ Velocity max = {v_max:.6f}")
# print(f"✅ RTM abs max = {y_max:.6f}")

# ✅ Velocity max = 4.763645
# ✅ RTM abs max = 63738.917969