import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from physicsnemo.models.fno import FNO
import h5py
from skimage.metrics import structural_similarity as ssim
from matplotlib import colors
from scipy.interpolate import interp1d
import time
import sys
import torch.nn.functional as F
from groundwater.devito_op import GroundwaterModel
import h5py
import os

from models.ns_inversion import NSModel  # Your model
from utils import get_dataset, load_config, get_model  # Your utils



# ----------------------
# Gaussian Smoothing Functions
# ----------------------
def gaussian_kernel(size: int, sigma: float):
    """Creates a 2D Gaussian kernel."""
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel = gauss[:, None] @ gauss[None, :]
    return kernel

def apply_gaussian_smoothing(batch_matrix: torch.Tensor, kernel_size: int, sigma: float):
    """Applies Gaussian smoothing to a batch of input matrices using a Gaussian kernel."""
    kernel = gaussian_kernel(kernel_size, sigma).to(batch_matrix.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: 1 x 1 x k x k
    kernel = kernel.expand(1, 1, kernel_size, kernel_size)

    original_min = batch_matrix.amin(dim=(-2, -1), keepdim=True)
    original_max = batch_matrix.amax(dim=(-2, -1), keepdim=True)
    original_range = original_max - original_min

    smoothed_batch = F.conv2d(batch_matrix, kernel, padding=kernel_size // 2, groups=1)
    smoothed_min = smoothed_batch.amin(dim=(-2, -1), keepdim=True)
    smoothed_max = smoothed_batch.amax(dim=(-2, -1), keepdim=True)
    smoothed_range = smoothed_max - smoothed_min

    rescaled_batch = (smoothed_batch - smoothed_min) / (smoothed_range + 1e-8) * original_range + original_min
    return rescaled_batch


laplacian_kernel = torch.tensor([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape [1,1,3,3]

def gradient_penalty(x):
    x = x.unsqueeze(1) if x.ndim == 3 else x  # ensure shape [B,1,H,W]
    weight = laplacian_kernel.to(x.device)
    lap = F.conv2d(x, weight, padding=1)
    return torch.mean(lap**2)


# ----------------------
# Plotting Functions
# ----------------------
def plot_single(true1, path, cmap="jet", vmin=None, vmax=None):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})
    print("vmin", vmin, vmax)
    if vmin != 0:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if (vmin is not None and vmax is not None) else colors.CenteredNorm()
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax) if (vmin is not None and vmax is not None) else colors.CenteredNorm()
    
    fig, ax = plt.subplots()
    cax = ax.imshow(true1, cmap=cmap, norm=norm)
    plt.colorbar(cax, ax=ax, fraction=0.045, pad=0.06)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_observed_only_with_scatter(data, x_idx, y_idx, ax, cmap='jet'):
    # extract the value at each observation
    vals = data[y_idx, x_idx]

    # choose same norm logic
    vmin, vmax = vals.min(), vals.max()
    if vmin < 0 < vmax:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    sc = ax.scatter(x_idx, y_idx, c=vals, cmap=cmap, norm=norm, s=50, marker='s')
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    return sc


def plot_inversion_result(x0, x, true_y, y, x_pred, loss_type, index, x_idx, y_idx):
    # pull everything off‐GPU, to numpy:
    fields = [
        x0.detach().squeeze().cpu().numpy(),       # initial guess
        true_y.squeeze().cpu().numpy(),            # ground truth output
        x.squeeze().cpu().numpy(),                 # ground truth input
        y.squeeze().cpu().numpy(),                 # forward prediction
        x_pred.detach().squeeze().cpu().numpy(),    # inversion result
        np.abs(x.squeeze().cpu().numpy() - x_pred.detach().squeeze().cpu().numpy())
    ]
    titles = [
        r'Initial Guess ($a_0$)',
        r'Ground Truth Output ($u$)',
        r'Ground Truth Input ($a^\ast$)',
        r'Forward Prediction ($\hat{u}$)',
        r'Inversion Result ($a$)',
        r'$|a - a^\ast|$'
    ]

    # your observed locations
    # if sub_sampling == False:
    #     x_idx = cols[:,0].long().cpu().numpy()
    #     y_idx = cols[:,1].long().cpu().numpy()
    # else:
    x_idx = x_idx.detach().cpu().numpy()
    y_idx = y_idx.detach().cpu().numpy()

    fig, axes = plt.subplots(3, 2, figsize=(10,15))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        data = fields[i]
        # choose norm
        vmin, vmax = data.min(), data.max()
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        if i in (1, 3):  # only observed points
            if sub_sampling == True:
                sc = ax.scatter(x_idx, y_idx, c=data[y_idx, x_idx], cmap='jet', norm=norm, s=10, marker='o')
            else:
                sc = ax.scatter(x_idx, y_idx, c=data[y_idx, x_idx], cmap='jet', norm=norm, s=5, marker='o')
            mappable = sc
            # set the axes limits to match the image‐grid
            ax.set_xlim(-0.5, data.shape[1]-0.5)
            ax.set_ylim(data.shape[0]-0.5, -0.5)      # flip y so origin matches imshow
            ax.set_aspect('equal')

        else:  # full‐field image
            im = ax.imshow(
                data,
                cmap='jet' if i<5 else 'magma',
                norm=norm,
                origin='lower',
                extent=(0, data.shape[1], 0, data.shape[0]),
                aspect='equal'
            )
            mappable = im

        ax.set_title(titles[i])
        # exactly one colorbar:
        fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if loss_type == "JAC" and top_subsampling == False:
        plt.savefig(f"inversion_result_{loss_type}_{num_vec}_{initial_guess}/inversion_result_{loss_type}_{index}.png")
    elif loss_type == "JAC" and top_subsampling == True:
        plt.savefig(f"inversion_result_{loss_type}_{num_vec}_{initial_guess}_top/inversion_result_{loss_type}_{index}.png")
    elif loss_type != "JAC" and top_subsampling == False:
        plt.savefig(f"inversion_result_{loss_type}_{initial_guess}/inversion_result_{loss_type}_{index}.png")
    else:
        plt.savefig(f"inversion_result_{loss_type}_{initial_guess}_top/inversion_result_{loss_type}_{index}.png")
    plt.close(fig)



def plot_ula_summary(posterior_set, true_x, output_dir=".", sample_index=0):
    posterior_tensor = torch.stack(posterior_set)  # shape: (num_samples, H, W)
    mean_map = posterior_tensor.mean(dim=0).squeeze()
    std_map = posterior_tensor.std(dim=0).squeeze()
    abs_error = (mean_map - true_x.squeeze().detach().cpu()).abs()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes[0].imshow(mean_map.cpu().numpy(), cmap="jet")
    axes[0].set_title("Posterior Mean")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(std_map.cpu().numpy(), cmap="viridis")
    axes[1].set_title("Uncertainty (Std)")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(abs_error.cpu().numpy(), cmap="magma")
    axes[2].set_title("Absolute Error")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ula_summary_sample_{sample_index}.png")
    plt.close()


def plot_ula_slice(posterior_set, true_x, slice_idx=64, output_dir=".", sample_index=0):
    posterior_tensor = torch.stack(posterior_set)
    samples_np = posterior_tensor[:, slice_idx, :].cpu().numpy()

    x = np.arange(samples_np.shape[1])
    mean_slice = samples_np.mean(axis=0)
    std_slice = samples_np.std(axis=0)
    lower = mean_slice - 1.96 * std_slice
    upper = mean_slice + 1.96 * std_slice
    truth = true_x.squeeze().cpu().numpy()[slice_idx]

    plt.figure(figsize=(10, 5))
    plt.fill_between(x, lower, upper, alpha=0.3, label="95% CI")
    plt.plot(x, mean_slice, label="Posterior Mean", color="blue")
    plt.plot(x, truth, "--", label="True", color="black")
    plt.title(f"Posterior Slice at y={slice_idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ula_slice_sample_{sample_index}.png")
    plt.close()


def plot_ula_trace(posterior_set, pix_y=32, pix_x=32, output_dir=".", sample_index=0):
    posterior_tensor = torch.stack(posterior_set)
    values = posterior_tensor[:, pix_y, pix_x].cpu().numpy()

    plt.figure(figsize=(8, 3))
    plt.plot(values)
    plt.title(f"Trace Plot at Pixel ({pix_y},{pix_x})")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ula_trace_pixel_sample_{sample_index}.png")
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.hist(values, bins=50, density=True)
    plt.title(f"Posterior Histogram at ({pix_y}, {pix_x})")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ula_hist_pixel_sample_{sample_index}.png")
    plt.close()

# Replaces least_squares_posterior_estimation with Metropolis-Adjusted Langevin Algorithm (MALA)

# def mala_sampling(model, y_obs, init_guess, prior, cols,
#                   num_samples=1000, step_size=1e-3, alpha=0.0):
#     """
#     MALA for Bayesian inversion using a surrogate model.

#     Args:
#         model: forward surrogate (e.g., FNO)
#         y_obs: observed data
#         init_guess: starting guess (torch.Tensor)
#         prior: prior sample (torch.Tensor)
#         cols: list of observed indices (N, 2)
#         num_samples: number of samples to draw
#         step_size: Langevin step size (epsilon)
#         alpha: regularization weight

#     Returns:
#         posterior_set: list of accepted samples
#         i, j: observation indices (torch.LongTensor)
#     """
#     def log_posterior(a):
#         print("a shape", a.shape)
#         # output = model(a)
#         # pred = output[i, j]
#         # target = y_obs[:, :, i, j]
#         if loss_type == "Devito":
#             squeezed_x0 = a.squeeze()
#             squeezed_x0.retain_grad()
#             output = model(squeezed_x0)
#         else:
#             output = model(a)

#         # extract and compute loss
#         if loss_type == "Devito":
#             extracted_output = output[i, j]
#             extracted_target = y_obs[:, :, i, j]
#         else:
#             extracted_output = output[:, :, i, j]
#             extracted_target = y_obs[:, :, i, j]
#             print("extracted output", extracted_output.shape)

#         neg_log_likelihood = mse(extracted_output.squeeze(), extracted_target.squeeze())

#         # neg_log_likelihood = mse(pred, target)
#         neg_log_prior = torch.norm(a - prior)
#         reg_term = alpha * gradient_penalty(a)
#         return -(neg_log_likelihood + neg_log_prior + reg_term)

#     def transition_log_prob(x_from, x_to, grad_from):
#         mu = x_from + 0.5 * step_size**2 * grad_from
#         return -((x_to - mu) ** 2).sum() / (2 * step_size**2)

#     a = init_guess.clone().detach().to(y_obs.device)
#     a.requires_grad_(True)
#     posterior_set = []
#     accepted = 0

#     i, j = cols[:, 0].long(), cols[:, 1].long()
#     mse = torch.nn.MSELoss()

#     for t in range(num_samples):
#         # Current log posterior and gradient
#         lp = log_posterior(a)
#         a.grad = None
#         lp.backward()
#         grad_a = a.grad.clone()

#         # Proposal
#         noise = torch.randn_like(a)
#         proposal = a + 0.5 * step_size**2 * grad_a + step_size * noise
#         proposal = proposal.clamp(0.0, 1.0).detach().requires_grad_(True)
#         proposal.retain_grad()

#         # Evaluate posterior at proposal
#         lp_prop = log_posterior(proposal)
#         proposal.grad = None
#         lp_prop.backward()
#         grad_prop = proposal.grad.clone()

#         # Transition log probabilities
#         log_q_forward = transition_log_prob(a, proposal, grad_a)
#         log_q_reverse = transition_log_prob(proposal, a, grad_prop)

#         log_accept_ratio = lp_prop - lp + log_q_reverse - log_q_forward
#         if torch.log(torch.rand(1)).item() < log_accept_ratio.item():
#             a = proposal.clone().detach().requires_grad_(True)
#             accepted += 1

#         posterior_set.append(a.detach().cpu())

#         if t % 100 == 0:
#             print(f"[Step {t}] log_posterior={lp.item():.4f}, Accepted={accepted}/{t+1}")

#     print(f"Final acceptance rate: {accepted / num_samples:.2f}")
#     return posterior_set, i, j

def mala_sampling(model, y_obs, init_guess, prior, cols,
                 num_samples=1000, step_size=1e-3, alpha=0.0):
    """
    Unadjusted Langevin Algorithm (ULA) for Bayesian inversion.

    Args:
        model: forward surrogate (e.g., FNO)
        y_obs: observed data
        init_guess: starting guess (torch.Tensor)
        prior: prior sample (torch.Tensor)
        cols: list of observed indices (N, 2)
        num_samples: number of samples to draw
        step_size: Langevin step size (epsilon)
        alpha: regularization weight

    Returns:
        posterior_set: list of ULA samples
        i, j: observation indices (torch.LongTensor)
    """
    def log_posterior(a):
        if loss_type == "Devito":
            squeezed_x0 = a.squeeze()
            squeezed_x0.retain_grad()
            output = model(squeezed_x0)
        else:
            output = model(a)

        if loss_type == "Devito":
            extracted_output = output[i, j]
            extracted_target = y_obs[:, :, i, j]
        else:
            extracted_output = output[:, :, i, j]
            extracted_target = y_obs[:, :, i, j]

        neg_log_likelihood = mse(extracted_output.squeeze(), extracted_target.squeeze())
        neg_log_prior = torch.norm(a - prior)
        reg_term = alpha * gradient_penalty(a)
        return -(neg_log_likelihood + neg_log_prior + reg_term)

    # Observation indices
    i, j = cols[:, 0].long(), cols[:, 1].long()
    mse = torch.nn.MSELoss()

    a = init_guess.clone().detach().to(y_obs.device).requires_grad_(True)
    posterior_set = []

    for t in range(num_samples):
        lp = log_posterior(a)
        a.grad = None
        lp.backward()
        grad = a.grad.clone()

        # ULA update (no Metropolis correction)
        noise = torch.randn_like(a)
        a = (a + 0.5 * step_size**2 * grad + step_size * noise).clamp(0.0, 1.0).detach().requires_grad_(True)

        posterior_set.append(a.detach().cpu())

        if t % 100 == 0:
            print(f"[Step {t}] log_posterior={lp.item():.4f}")

    return posterior_set, i, j



# ----------------------
# Main Script for Inversion on Multiple Samples (batch_size=1)
# ----------------------
if __name__ == "__main__":
    # Set up device and random seed.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print(f"Using device: {device}")

    # Define simulation parameters.
    num_vec = 100
    loss_type = "JAC"  # or "JAC" "MSE" "Devito"
    GRF = 2
    alpha = 0.05
    noise_std = 0.4
    initial_guess = "smooth" # "smooth", "noisy"
    sub_sampling = False
    top_subsampling = True
    full_obs = False

    if initial_guess == "prior_mean":
        learning_rate = 0.001 # 0.0001 (grf, fullobs) #0.005 (noisy, fullobs) #0.00005  # Inversion learning rate.
        num_sample = 10 #1
        num_sample_prior = 100 #5
        num_epoch = 2001 #1001
        offset=130
    elif initial_guess == "smooth":
        learning_rate = 0.0001 # 0.0001 (grf, fullobs) #0.005 (noisy, fullobs) #0.00005  # Inversion learning rate.
        num_sample = 3
        num_sample_prior = 100
        num_epoch = 2001 #1001
        offset=120
        if GRF == 1:
            kernel_size = 45 #55 #(grf, fullobs)
            sigma = 10.0 #100.0 # (grf, fullobs)
        elif GRF == 2:
            kernel_size = 55
            sigma = 100.0


    
    
    # Load configuration and dataset. and checkpoint
    if loss_type == "JAC" and num_vec == 1:
        config = "configs/eigenvectors/e=1.yaml"
        ckpt_path = "checkpoints/n=128_e=1_m=FNO_s=RFS_l=JAC_20250513_164312/n=128_e=1_m=FNO_s=RFS_l=JAC_epoch=249_val_rel_l2_loss=0.0006.ckpt"
    elif loss_type == "JAC" and num_vec == 10:
        config = "output/n=128_e=10_m=FNO_s=RFS_l=JAC_20250512_144619/config.yaml"
        ckpt_path = "checkpoints/n=128_e=10_m=FNO_s=RFS_l=JAC_20250512_144619/n=128_e=10_m=FNO_s=RFS_l=JAC_epoch=249_val_rel_l2_loss=0.0005.ckpt"
    elif loss_type == "JAC" and num_vec == 50:
        # config = load_config(f"output/n=128_e=8_m=FNO_s=RFS_l=JAC_lamba=0.5_20250421_221953/config.yaml")
        # config = load_config("output/Darcy_training_20250507_175531/config.yaml")
        config = load_config("output/n=128_e=50_m=FNO_s=RFS_l=JAC_20250512_141821/config.yaml")
        # ckpt_path = f"checkpoints/n=128_e=8_m=FNO_s=RFS_l=JAC_lamba=0.5_20250421_221953/last.ckpt"
        # ckpt_path = f"checkpoints/Darcy_training_20250507_175531/Darcy_training_epoch=149_val_rel_l2_loss=0.0082.ckpt"
        ckpt_path = f"checkpoints/n=128_e=50_m=FNO_s=RFS_l=JAC_20250514_151731/n=128_e=50_m=FNO_s=RFS_l=JAC_epoch=249_val_rel_l2_loss=0.0004.ckpt"
        # DeFINO_Richard/train/checkpoints/n=128_e=50_m=FNO_s=RFS_l=JAC_20250514_151731/n=128_e=50_m=FNO_s=RFS_l=JAC_epoch=249_val_rel_l2_loss=0.0004.ckpt
    elif loss_type == "JAC" and num_vec == 100:
        config = load_config("configs/eigenvectors/e_100.yaml")
        ckpt_path = f"checkpoints/DARCY_JAC_100/Darcy_training_epoch=249_val_rel_l2_loss=0.0022_JAC_May14.ckpt"
    elif loss_type == "RAND":
        config = load_config("output/n=128_e=8_m=FNO_s=RAND_l=JAC_20250421_124311/config.yaml")
        ckpt_path = f"checkpoints/n=128_e=8_m=FNO_s=RAND_l=JAC_20250421_125959/last.ckpt"
    elif loss_type == "MSE":
        config = load_config("configs/darcy_MSE.yaml")
        ckpt_path = "checkpoints/DARCY_MSE/Darcy_training_epoch=249_val_rel_l2_loss=0.0009_MSE_May14.ckpt"
    
    if loss_type != "Devito":
        model = NSModel.load_from_checkpoint(ckpt_path).eval().to(device)

    # Load Data
    data_config = load_config("output/n=128_e=50_m=FNO_s=RFS_l=JAC_20250512_141821/config.yaml")
    dataset = get_dataset(data_config.experiment.dataset_type, data_config.data_settings)
    dataloader = dataset.get_dataloader(offset=offset, limit=num_sample)
    prior_dataloader = dataset.get_dataloader(offset=offset, limit=num_sample_prior)

    # Initialize a list to hold loss and metric data for each sample.
    loss_data_all = []
    metrics_all = []
    sample_counter = 0
    final_ssim_list = []
    final_l2_list = []


    if loss_type == "JAC" and top_subsampling == False :
        fname = f'inversion_history_{loss_type}_{num_vec}_{initial_guess}.h5'
    elif loss_type == "JAC" and top_subsampling == True:
        fname = f'inversion_history_{loss_type}_{num_vec}_{initial_guess}_top.h5'
    elif loss_type != "JAC" and top_subsampling == False :
        fname = f'inversion_history_{loss_type}_{initial_guess}.h5'
    else:
        fname = f'inversion_history_{loss_type}_{initial_guess}_top.h5'
    # If it already exists, delete it (and any stale lock)
    if os.path.exists(fname):
        os.remove(fname)

    # Now create it
    h5_file = h5py.File(fname, 'w')
    num_samples = len(dataloader)
    dset = h5_file.create_dataset(
        'a', 
        shape=(num_samples, num_epoch, 128, 128),
        dtype='f4',
        compression='gzip',
        compression_opts=4,
        chunks=(1, num_epoch, 128, 128)  # chunk by sample
    )

    # Compute prior mean
    if initial_guess == "prior_mean":
        sum_x = 0.0
        n_samples = 0
        for batch in prior_dataloader:
            x = batch['x'].to(device)
            sum_x += x.squeeze()
            n_samples += 1

        print("Prior averaged over ", n_samples)
        prior_mean = sum_x / n_samples  # shape: [C, H, W]
        prior_mean = prior_mean.unsqueeze(dim=0).unsqueeze(dim=1).detach()

    # Prepare CSV accumulators:
    loss_data_all = []
    sample_counter = 0

    for batch in dataloader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        L = batch['L'].view(-1).to(device)
        d = int(x.shape[-1])
        cols = torch.tensor([ (idx.item() // d, idx.item() % d) for idx in L ], device=device)

        # initial guess logic …
        if initial_guess == "smooth":
            zero_X = apply_gaussian_smoothing(x, kernel_size, sigma) + 1e-3
        elif initial_guess == "noisy":
            zero_X = x + torch.randn_like(x) * noise_std
        elif initial_guess == "prior_mean":
            zero_X = prior_mean
        plot_single(zero_X.detach().cpu().squeeze(), f"zero_X_sample_{sample_counter}.png", "jet")

        if loss_type == "Devito":
            forcing_term = torch.zeros(zero_X.squeeze().shape)
            groundwater_model = GroundwaterModel(forcing_term.shape[0])
            model = lambda x: groundwater_model(x, forcing_term)

        posterior_set, i_idx, j_idx = mala_sampling(
            model=model,
            y_obs=y,
            init_guess=zero_X.clone().detach(),
            prior=x,
            cols=cols,
            num_samples=num_epoch,
            step_size=learning_rate,
            alpha=alpha
        )

        true_x = x  # (1,1,H,W) or (1,H,W)
        plot_ula_summary(posterior_set, true_x, output_dir=".", sample_index=sample_counter)
        plot_ula_slice(posterior_set, true_x, slice_idx=64, output_dir=".", sample_index=sample_counter)
        plot_ula_trace(posterior_set, pix_y=32, pix_x=32, output_dir=".", sample_index=sample_counter)



        # Plot the final inversion result.
        final_x0 = torch.tensor(posterior_set[-1]).detach()
        # plot_inversion_result(zero_X, x, y, pred, final_x0, loss_type, sample_counter, i_idx, j_idx)

        # posterior_set is a list of length num_epoch, each an 128×128 numpy array.
        # Write them into the HDF5 at [sample_counter, :, :, :]:
        arr = np.stack(posterior_set, axis=0).squeeze()   # shape (num_epoch,128,128)
        dset[sample_counter, :, :, :] = arr

        # collect this sample’s iteration‐by‐iteration records
        loss_data_all.extend(loss_data_iter)
        sample_counter += 1

    # save to single CSV
    df = pd.DataFrame(loss_data_all)
    # Close the HDF5 file when you’re done:
    h5_file.close()
    with h5py.File(fname, 'r') as f:
        print("On‑disk dataset shape is", f['a'].shape)

    # # Compute and print averaged SSIM and L2 misfit over all samples.
    # average_ssim = np.mean(final_ssim_list)
    # average_l2 = np.mean(final_l2_list)
    # print(f"\nAveraged Final SSIM over {sample_counter} sample(s): {average_ssim:.4f}")
    # print(f"Averaged Final Relative L2 misfit over {sample_counter} sample(s): {average_l2:.4f}")
    # # @TODO I want to save it in some file.

    # # Save all loss and metric data to CSV.
    # df = pd.DataFrame(loss_data_all)
    # df_min = pd.DataFrame(metrics_all)
    # print(df_min)

    if loss_type == "JAC" and top_subsampling == False:
        csv_file = f"loss_statistics_multiple_samples_{loss_type}_{num_vec}_{initial_guess}.csv"
    elif loss_type == "JAC" and top_subsampling == True:
        csv_file = f"loss_statistics_multiple_samples_{loss_type}_{num_vec}_{initial_guess}_top.csv"
    elif loss_type != "JAC" and top_subsampling == False:
        csv_file = f"loss_statistics_multiple_samples_{loss_type}_{initial_guess}.csv"
    else:
        csv_file = f"loss_statistics_multiple_samples_{loss_type}_{initial_guess}_top.csv"

    df.to_csv(csv_file, index=False)
    print(f"Loss data saved to {csv_file}")