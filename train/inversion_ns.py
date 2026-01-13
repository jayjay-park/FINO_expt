#!/usr/bin/env python3
import os, time, datetime, sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import load_config, get_dataset
from utils_inversion import *
import matplotlib.colors as mcolors
import torch.fft as tfft
from utils import load_config, get_dataset
from models.ns_inversion import NSModel

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datasets.simulators.NS import NavierStokesSimulator

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
data_type = "NS"
forward_type = "mse" #[fino, numerical, mse]
num_vec = 400
grid_N = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# multiscale / TV hyperparameters
use_multiscale = True
ms_sigmas = [6.0, 3.0, 1.0, 0.0]   # Gaussian σ for 64×64 grid
ms_weights = [0.05, 0.1, 0.3, 1.0]
use_tv_prior = False
alpha_tv = 1e-6
eps_tv = 1e-6

# result folder (timestamped)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"./inverse_ns_multiscale_{forward_type}_{timestamp}"
os.makedirs(out_dir, exist_ok=True)
print(f"Results will be saved in: {out_dir}")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_bchw(x):
    # Always reshape to [B,C,H,W] by flattening leading dims into B
    if x.dim() == 4:
        return x
    elif x.dim() > 4:
        # assume last three are H,W (with C before them)
        C, H, W = x.shape[-3], x.shape[-2], x.shape[-1]
        B = int(x.numel() // (C*H*W))
        return x.contiguous().view(B, C, H, W)
    else:
        raise ValueError(f"Expected ≥4D, got {x.shape}")

def gaussian_blur_2d(x: torch.Tensor, sigma: float):
    x = _ensure_bchw(x)
    if sigma <= 0:
        return x
    B, C, H, W = x.shape
    k1 = _gaussian_kernel_1d(sigma, x.device, x.dtype)
    ky = k1.view(1,1,-1,1).repeat(C,1,1,1)
    kx = k1.view(1,1,1,-1).repeat(C,1,1,1)
    pad_y = ky.shape[2] // 2
    pad_x = kx.shape[3] // 2
    x = F.pad(x, (pad_x, pad_x, pad_y, pad_y), mode='reflect')  # or 'circular'
    x = F.conv2d(x, ky, padding=0, groups=C)
    x = F.conv2d(x, kx, padding=0, groups=C)
    return x

def grf_quadratic_prior(x, alpha=2.5, tau=7.0):
    x = _ensure_bchw(x)                               # <— important
    B, C, H, W = x.shape
    ky = tfft.fftfreq(H, d=1.0).to(x.device).reshape(H,1)
    kx = tfft.fftfreq(W, d=1.0).to(x.device).reshape(1,W)
    K2 = (2*torch.pi)**2 * (ky**2 + kx**2)
    lam = (K2 + tau**2).pow(alpha/2)
    X = tfft.fft2(x.squeeze(1))                       # [B,H,W], C=1
    penalty = (lam * (X.abs()**2)).sum(dim=(-2,-1)) / (H*W)
    return 0.5 * penalty.mean()



def _gaussian_kernel_1d(sigma, device, dtype):
    if sigma <= 0:
        return torch.tensor([1.0], device=device, dtype=dtype)
    half = int(round(3 * sigma))
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    k = torch.exp(-(x**2) / (2 * sigma * sigma))
    return k / k.sum()


def tv_loss_nchw(x: torch.Tensor, eps: float = 1e-6, clip_grad: float = 1e3) -> torch.Tensor:
    """
    Stable isotropic total variation loss.
    """
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = x[:, :, :, 1:] - x[:, :, :, :-1]

    # crop to common inner region
    dx = dx[:, :, :, :-1]
    dy = dy[:, :, :-1, :]

    # clip huge finite differences to prevent overflow
    dx = torch.clamp(dx, -clip_grad, clip_grad)
    dy = torch.clamp(dy, -clip_grad, clip_grad)

    grad_mag = torch.sqrt(dx * dx + dy * dy + eps)
    return grad_mag.mean()



def plot_field(field, title, filename):
    arr = field.squeeze().detach().cpu().numpy()
    plt.figure(figsize=(4,4))
    plt.imshow(arr, cmap="RdBu_r", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL (numerical simulator)
# ─────────────────────────────────────────────────────────────────────────────
if forward_type == "fino" and num_vec == 200:
    config = load_config("configs/eigenvectors/e_400_NS_new.yaml")
    ckpt_path = f"checkpoints/n=1000_e=200_m=FNO_s=RFS_l=JAC_20250903_013609/n=1000_e=200_m=FNO_s=RFS_l=JAC_epoch=200_val_rel_l2_loss=0.1504.ckpt"
elif forward_type == "fino" and num_vec == 400:
    config = load_config("configs/eigenvectors/e_400_NS_new.yaml")
    # ckpt_path = f"checkpoints/n=400_e=400_m=FNO_s=RFS_l=JAC_20250907_142507/n=400_e=400_m=FNO_s=RFS_l=JAC_epoch=183_val_rel_l2_loss=0.1652.ckpt"
    ckpt_path = "checkpoints/n=400_e=400_m=FNO_s=RFS_l=JAC_20250907_142507/n=400_e=400_m=FNO_s=RFS_l=JAC_epoch=170_val_rel_l2_loss=0.1650.ckpt"
elif forward_type == "mse":
    config = load_config("configs/eigenvectors/e_0_NS_new.yaml")
    ckpt_path = "checkpoints/n=1000_m=FNO_l=L2_20250915_114551/n=1000_m=FNO_l=L2_epoch=122_val_rel_l2_loss=0.1655.ckpt"
elif forward_type == "numerical":
    model = NavierStokesSimulator(grid_N, grid_N, 200, 10.0, 1e-3).to(device)
    model.eval()

def model_forward(init_field):
    """Wrapper consistent with your inversion code."""
    if forward_type == "numerical":
        return model(init_field, T_final=10.0).eval()
    else:
        model = NSModel.load_from_checkpoint(ckpt_path).eval().to(device)
        return model(init_field)

# ─────────────────────────────────────────────────────────────────────────────
# DATALOADING (your repository style)
# ─────────────────────────────────────────────────────────────────────────────

if data_type == "NS":
    data_config = load_config("configs/eigenvectors/e_400_NS_new.yaml")

# inference dataset
num_sample = 1
num_sample_prior = 1

data_config.data_settings["batch_size"] = 1
dataset = get_dataset(data_config.experiment.dataset_type, data_config.data_settings)
dataloader = dataset.get_dataloader(offset=405, limit=num_sample)

# prior dataset
data_config.data_settings["batch_size"] = 50
dataset = get_dataset(data_config.experiment.dataset_type, data_config.data_settings)
prior_dataloader = dataset.get_dataloader(offset=414, limit=num_sample_prior)

# extract sample (assuming same structure)
sample = next(iter(dataloader))
true_vorticity = sample["y"].unsqueeze(0).to(device)  # [1,1,64,64]
x = sample["x"].to(device)
x0 = apply_gaussian_smoothing(x.detach().clone(), kernel_size=33, sigma=30)
plot_field(true_vorticity.squeeze(), "True vorticity (T=10)", "true_vorticity.png")
plot_field(x.squeeze(), "True vorticity (T=0)", "true_initial_vorticity.png")
plot_field(x0.squeeze(), "Initial Guess", "initial_guess.png")

# ─────────────────────────────────────────────────────────────────────────────
# INVERSION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def inverse_ns_leastsquares(model_forward, x0, y_target, T_final=10.0, lr=5e-3, n_iter=1000, forward_type="numerical"):
    # 1) cast FIRST (and match y_target)
    if forward_type == "numerical":
        x0 = x0.to(dtype=torch.float64)
        y_target = y_target.to(dtype=torch.float64)

    # 2) detach→clone so it's a LEAF, then require grad
    x0 = x0.detach().clone().requires_grad_(True)
    # 3) create the optimizer AFTER x0 is finalized
    optimizer = torch.optim.Adam([x0], lr=lr)
    losses = []
    alpha_grf = 1e-7  # tune in [1e-4, 1e-2]
    assert optimizer.param_groups[0]['params'][0] is x0, "Optimizer is not updating the x0 used in forward."


    start = time.time()
    for it in range(n_iter):
        optimizer.zero_grad()
        print("shape", x0.shape)
        pred = model_forward(x0)

        # multiscale MSE
        if use_multiscale:
            loss_data = 0.0
            for w, sig in zip(ms_weights, ms_sigmas):
                pred_s = gaussian_blur_2d(pred, sig)
                tgt_s  = gaussian_blur_2d(y_target, sig)
                loss_data += w * F.mse_loss(pred_s, tgt_s, reduction="mean")
        else:
            loss_data = F.mse_loss(pred, y_target, reduction="mean")


        # TV regularization
        if use_tv_prior:
            loss_prior = alpha_tv * tv_loss_nchw(x0) if use_tv_prior else 0.0
        else:
            loss_prior = alpha_grf * grf_quadratic_prior(x0, alpha=2.5, tau=7.0)
        loss = loss_data + loss_prior
        loss.backward()
        optimizer.step()

        if it == 0:
            g = x0.grad.detach()
            plot_field(g, "Gradient wrt x0 at iter 0", "grad_x0_iter0.png")


        losses.append(loss.item())
        gn = torch.nan_to_num(x0.grad).norm().item()
        if it % 5 == 0:
            plot_field(x0.clone().detach().cpu(), f"Inverted at {it}", f"reconstructed_{it}.png")
            print(f"[{it:04d}] total={loss.item():.3e}, data={loss_data.item():.7e}, ||g|| = {gn:.4e}, tv={loss_prior.item() if not isinstance(loss_prior,float) else loss_prior:.3e}")

    print(f"Total time: {time.time()-start:.1f}s")
    return x0.detach(), losses

# ─────────────────────────────────────────────────────────────────────────────
# RUN INVERSION
# ─────────────────────────────────────────────────────────────────────────────
print("Initial field stats:", x0.min().item(), x0.max().item(), x0.std().item())

x_recon, losses = inverse_ns_leastsquares(model_forward, x0, true_vorticity, lr=8e-2, n_iter=200, forward_type=forward_type)
torch.save(x_recon, os.path.join(out_dir, "reconstructed_init.pt"))
torch.save(losses,  os.path.join(out_dir, "losses.pt"))

# forward simulate again for sanity check
pred_final = model_forward(x_recon)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
plot_field(x_recon, "Reconstructed initial field ω₀", "reconstructed_init.png")
plot_field(pred_final, "Predicted vorticity at T=10", "pred_vorticity.png")


norm = mcolors.CenteredNorm()
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(pred_final.detach().squeeze().cpu(), cmap="RdBu_r"); plt.title("Predicted ω(T=10)")
plt.subplot(1,3,2)
plt.imshow(true_vorticity.detach().squeeze().cpu(), cmap="RdBu_r"); plt.title("True ω(T=10)")
plt.subplot(1,3,3)
plt.imshow(true_vorticity.detach().squeeze().cpu() - true_vorticity.detach().squeeze().cpu(), cmap="RdGy", norm=norm); plt.title("True ω(T=10)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "comparison.png"))
plt.close()

plt.figure()
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Least Squares Loss (Multiscale)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "loss_curve.png"))
plt.close()

print("✅ Inversion complete. Results saved in", out_dir)
