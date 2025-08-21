# #!/usr/bin/env python3
# """make_movie.py

# Make a quick GIF **or** MP4 that visualises how a 2-D field evolves over the
# iterations stored in an HDF5 trajectory.

# ✔ Works with recent Matplotlib (≥ 3.8) – uses ``fig.savefig`` to an in-memory
#   PNG instead of the deprecated ``tostring_rgb`` method.
# ✔ Only needs *h5py*, *imageio* and *matplotlib* (plus FFmpeg for MP4).

# Example
# -------
# ::

#    python make_movie.py prior_mean_wonoise/inversion_history_JAC_400_prior_mean.h5 \
#        field_evolution.gif --dataset a --fps 12 --cmap viridis

# Passing an output name ending in ``.mp4`` triggers the H.264 writer instead of
# GIF when FFmpeg is available.
# """

# from __future__ import annotations

# import argparse
# import io
# from pathlib import Path
# from typing import Iterable, List

# import h5py
# import imageio.v2 as imageio  # pillow backend is fine
# import matplotlib.pyplot as plt
# import numpy as np


# # -----------------------------------------------------------------------------
# # I/O helpers
# # -----------------------------------------------------------------------------

# def load_trajectory(h5_path: Path, dataset_key: str) -> np.ndarray:
#     """Return the trajectory as ``(T, H, W)`` float array."""
#     with h5py.File(h5_path, "r") as f:
#         data = f[dataset_key][:]
#     data = np.squeeze(data)  # drop length-1 axes like (1,T,H,W)
#     if data.ndim != 3:
#         raise ValueError(
#             f"Unexpected shape {data.shape}; expecting (T, H, W) or (1, T, H, W)."
#         )
#     return data


# def build_frames(
#     trajectory: np.ndarray,
#     cmap: str,
#     vmin: float | None = None,
#     vmax: float | None = None,
#     colorbar: bool = False,
# ) -> List[np.ndarray]:
#     """Render each 2-D slice to an RGB image (returns list of numpy arrays)."""
#     vmin = np.min(trajectory) if vmin is None else vmin
#     vmax = np.max(trajectory) if vmax is None else vmax

#     frames: list[np.ndarray] = []
#     for k, field in enumerate(trajectory):
#         fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
#         im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_title(f"Iter {k}")
#         if colorbar:
#             fig.colorbar(im, fraction=0.046, pad=0.04)
#         fig.tight_layout(pad=0)

#         buf = io.BytesIO()
#         fig.savefig(buf, format="png", bbox_inches="tight")
#         plt.close(fig)
#         buf.seek(0)
#         frames.append(imageio.imread(buf))
#     return frames


# def write_movie(frames: Iterable[np.ndarray], output_path: Path, fps: int = 10) -> None:
#     """Write *frames* to GIF or MP4 based on file extension."""
#     ext = output_path.suffix.lower()
#     if ext == ".gif":
#         imageio.mimsave(output_path, frames, fps=fps)
#     elif ext == ".mp4":
#         with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8) as w:
#             for f in frames:
#                 w.append_data(f)
#     else:
#         raise ValueError("Output file must end with .gif or .mp4")


# # -----------------------------------------------------------------------------
# # CLI
# # -----------------------------------------------------------------------------

# def parse_args(argv: list[str] | None = None):
#     p = argparse.ArgumentParser(description="Create a field-evolution movie from an HDF5 trajectory.")
#     p.add_argument("h5", type=Path, help="Trajectory file (HDF5)")
#     p.add_argument("output", type=Path, help="Output GIF or MP4 filename")
#     p.add_argument("--dataset", default="a", help="Dataset key inside the HDF5 file (default: 'a')")
#     p.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
#     p.add_argument("--cmap", default="viridis", help="Matplotlib colormap (default: viridis)")
#     p.add_argument("--vmin", type=float, default=None, help="Fixed vmin for color scale")
#     p.add_argument("--vmax", type=float, default=None, help="Fixed vmax for color scale")
#     p.add_argument("--colorbar", action="store_true", help="Render a colorbar on each frame")
#     return p.parse_args(argv)


# def main(argv: list[str] | None = None) -> None:
#     args = parse_args(argv)

#     traj = load_trajectory(args.h5, args.dataset)
#     frames = build_frames(traj, args.cmap, args.vmin, args.vmax, args.colorbar)
#     write_movie(frames, args.output, args.fps)

#     print(f"Saved movie to {args.output.resolve()}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""make_movie.py — Create GIF/MP4 (no external FFmpeg required).

New fallback (2025‑07‑07 evening)
---------------------------------
* **OpenCV fallback**: if an external `ffmpeg` binary is missing, we now try
  `opencv-python(-headless)`’s built‑in encoder.  That means you can still get an
  **MP4** as long as `pip install opencv-python-headless` succeeds, even on
  clusters where you can’t write to the system.
* Behaviour order for `.mp4` output:
    1. external *ffmpeg* via *imageio* (highest quality, smallest file)
    2. built‑in OpenCV encoder (`cv2.VideoWriter`) – no external deps
    3. graceful fallback to `.gif` with a helpful message
* Everything else (interactive viewer, colour‑bar, etc.) unchanged.
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
from pathlib import Path
from typing import Iterable, List

import h5py
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

# Try importing cv2 for MP4 fallback
try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_trajectory(h5_path: Path, dataset_key: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        data = f[dataset_key][:]
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"Unexpected shape {data.shape}; expected (T,H,W) or (1,T,H,W)")
    return data


def build_frames(trajectory: np.ndarray, cmap: str, vmin: float | None, vmax: float | None, colorbar: bool) -> List[np.ndarray]:
    vmin = np.min(trajectory) if vmin is None else vmin
    vmax = np.max(trajectory) if vmax is None else vmax
    frames: list[np.ndarray] = []
    for k, field in enumerate(trajectory):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Iter {k}")
        if colorbar:
            fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout(pad=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))
    return frames


# -----------------------------------------------------------------------------
# Movie writing – ffmpeg → cv2 → gif
# -----------------------------------------------------------------------------

def _write_mp4_with_cv2(frames: List[np.ndarray], output: Path, fps: int):
    if not _CV2_AVAILABLE:
        raise RuntimeError("opencv-python not installed; cannot use cv2 fallback")
    h, w = frames[0].shape[:2]
    # Convert RGB → BGR and write
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output), fourcc, fps, (w, h))
    for f in frames:
        if f.shape[2] == 4:  # drop alpha if present
            f = f[..., :3]
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()


def write_movie(frames: Iterable[np.ndarray], output: Path, fps: int):
    ext = output.suffix.lower()

    # --- GIF ----------------------------------------------------
    if ext == ".gif":
        imageio.mimsave(output, frames, fps=fps)
        return

    # --- MP4 ----------------------------------------------------
    if ext == ".mp4":
        # (1) try external ffmpeg via imageio
        if shutil.which("ffmpeg") is not None:
            try:
                with imageio.get_writer(output, format="ffmpeg", fps=fps, codec="libx264", quality=8) as w:
                    for f in frames:
                        w.append_data(f)
                return
            except Exception as e:
                print(f"ffmpeg writer failed ({e}); trying OpenCV fallback…", file=sys.stderr)

        # (2) try OpenCV’s built‑in encoder
        try:
            _write_mp4_with_cv2(list(frames), output, fps)
            return
        except Exception as e:
            print(f"OpenCV writer failed ({e}); falling back to GIF…", file=sys.stderr)

        # (3) fall back to GIF
        alt = output.with_suffix(".gif")
        imageio.mimsave(alt, frames, fps=fps)
        sys.exit(f"Could not write MP4. GIF written instead: {alt}")

    raise ValueError("Output must end with .gif or .mp4")


# -----------------------------------------------------------------------------
# Interactive viewer (unchanged)
# -----------------------------------------------------------------------------

def interactive_viewer(trajectory: np.ndarray, cmap: str, vmin: float | None, vmax: float | None, colorbar: bool):
    vmin = np.min(trajectory) if vmin is None else vmin
    vmax = np.max(trajectory) if vmax is None else vmax
    n = trajectory.shape[0]

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.subplots_adjust(bottom=0.2)

    im = ax.imshow(trajectory[0], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Iter 0")
    if colorbar:
        fig.colorbar(im, fraction=0.046, pad=0.04)

    s_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(s_ax, "Iter", 0, n - 1, valinit=0, valstep=1)

    b_prev = Button(plt.axes([0.05, 0.1, 0.07, 0.04]), "<")
    b_next = Button(plt.axes([0.88, 0.1, 0.07, 0.04]), ">")

    def update(idx: int):
        im.set_data(trajectory[idx])
        ax.set_title(f"Iter {idx}")
        fig.canvas.draw_idle()

    slider.on_changed(lambda v: update(int(v)))
    b_prev.on_clicked(lambda _: slider.set_val((slider.val - 1) % n))
    b_next.on_clicked(lambda _: slider.set_val((slider.val + 1) % n))

    plt.show()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Create GIF/MP4 or interactively browse a trajectory in an HDF5 file.")
    p.add_argument("h5", type=Path, help="Input HDF5 trajectory")
    p.add_argument("output", type=Path, nargs="?", help="Output .gif or .mp4 (omit with --interactive)")
    p.add_argument("--dataset", default="a", help="HDF5 dataset key (default: a)")
    p.add_argument("--fps", type=int, default=10, help="Frames-per-second for video (default: 10)")
    p.add_argument("--cmap", default="viridis", help="Matplotlib colormap")
    p.add_argument("--vmin", type=float, help="Fixed vmin for colour scale")
    p.add_argument("--vmax", type=float, help="Fixed vmax for colour scale")
    p.add_argument("--colorbar", action="store_true", help="Display colour-bar")
    p.add_argument("--interactive", action="store_true", help="Launch interactive viewer instead of saving movie")
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse(argv)
    traj = load_trajectory(args.h5, args.dataset)

    if args.interactive:
        interactive_viewer(traj, args.cmap, args.vmin, args.vmax, args.colorbar)
        return

    if args.output is None:
        sys.exit("Error: output filename required unless --interactive is set")

    frames = build_frames(traj, args.cmap, args.vmin, args.vmax, args.colorbar)
    write_movie(frames, args.output, args.fps)
    print(f"Saved movie to {args.output.resolve()}")


if __name__ == "__main__":
    main()


# python plot_movie.py prior_mean_wonoise/inversion_history_JAC_400_prior_mean.h5 field_evolution.gif

# # 1 ) fast MP4 with colour-bar
# python plot_movie.py prior_mean_wonoise/inversion_history_JAC_400_prior_mean.h5 field_evolution.mp4 --dataset a --fps 20 --colorbar

# # 2 ) just poke around interactively (no file created)
# python make_movie.py prior_mean_wonoise/inversion_history_JAC_400_prior_mean.h5 \
#        --interactive --dataset a --cmap plasma --colorbar