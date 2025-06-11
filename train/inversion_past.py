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
    if sub_sampling != True:
        x_idx = cols[:,0].long().cpu().numpy()
        y_idx = cols[:,1].long().cpu().numpy()
    else:
        x_idx = x_idx.detach().cpu().numpy()
        y_idx = y_idx.detach().cpu().numpy()

    fig, axes = plt.subplots(3, 2, figsize=(10,15))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        data = fields[i]
        # choose norm
        vmin, vmax = data.min(), data.max()
        # if vmin < 0 < vmax:
        #     norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        # else:
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
    if loss_type == "JAC":
        plt.savefig(f"inversion_result_{loss_type}_{num_vec}_{initial_guess}/inversion_result_{loss_type}_{index}.png")
    else:
        plt.savefig(f"inversion_result_{loss_type}_{initial_guess}/inversion_result_{loss_type}_{index}.png")
    plt.close(fig)


# # ----------------------
# # Least Squares Posterior Estimation
# # ----------------------
# def least_squares_posterior_estimation(model, input_data, true_data, learning_rate, batch_num, num_iterations=500, prior=None):
#     if loss_type != "Devito":
#         model.eval()  # Freeze model parameters
#     mse_loss = torch.nn.MSELoss()

#     x0 = input_data.clone().detach().requires_grad_(True).to(device)
#     posterior_set = []
#     # Set boundaries based on the prior.
#     true_min = torch.min(prior) - 0.1
#     true_max = torch.max(prior) + 0.1
#     print("True range:", true_min.item(), true_max.item())

#     optimizer = torch.optim.Adam([x0], lr=learning_rate)
#     losses, inversion_MSEs, regs, ssims, metrics_per_minute = [], [], [], [], []
#     start_time = time.time()
#     last_record_time = start_time
#     global num_iter

#     plot_single(true_data.detach().cpu().squeeze(), f'true_data.png')
#     i = cols[:, 0].long()
#     j = cols[:, 1].long()

#     for iteration in range(num_iterations):
#         optimizer.zero_grad()
#         if loss_type == "Devito":
#             x0 = x0.squeeze()
#         output = model(x0)

#         if loss_type == "Devito":
#             extracted_output = output[i, j]
#             extracted_target = true_data[:, :, i, j]
#             print("extracted", extracted_output.shape, extracted_target.shape)
#             x0 = x0.reshape(1, 1, x0.shape[0], x0.shape[1])
#         else:
#             extracted_output = output[:, :, i, j]
#             extracted_target = true_data[:, :, i, j]

#         loss = mse_loss(extracted_output.squeeze(), extracted_target.squeeze())
#         # loss = mse_loss(output, true_data)
#         reg = gradient_penalty(x0)
#         print("reg", reg.item())

#         loss_total = loss + alpha * reg
#         loss_total.backward()
#         optimizer.step()

#         losses.append(loss_total.item()) # inversion_MSE = F.mse_loss(x0, prior)
#         diff = x0 - prior
#         inversion_MSE =     torch.norm(diff) / torch.norm(prior)   
#         inversion_MSEs.append(inversion_MSE.item())
#         regs.append(reg.item())
#         input_numpy = x0.detach().cpu().squeeze().numpy()
#         prior_numpy = prior.detach().cpu().squeeze().numpy()
#         ssim_value = ssim(input_numpy.astype(np.float64), prior_numpy.astype(np.float64),
#                           data_range=float(input_numpy.max()-input_numpy.min()))
#         ssims.append(ssim_value)

#         # check wall‐clock time
#         now = time.time()
#         print("now", now, start_time)
#         if now - last_record_time >= 10.0:
#             metrics_per_minute.append({
#                 'iteration': iteration,
#                 'elapsed_s': now - start_time,
#                 'loss': loss_total.item(),
#                 'inversion_MSE': inversion_MSE.item(),
#                 'regularization': reg.item(),
#                 'SSIM': ssim_value
#             })
#             last_record_time = now

#         if batch_num < 2 and iteration % 50 == 0 and loss_type != "Devito":
#             gradient = x0.grad.detach().cpu().squeeze()  # shape: [H, W] or similar
#             plt.imshow(gradient.numpy(), cmap='viridis')
#             plt.colorbar(label='Gradient Value', shrink=0.8)
#             plt.title('Gradient w.r.t. Input x0')
#             if loss_type == "JAC":
#                 plt.savefig(f'inversion_result_{loss_type}_{num_vec}_{initial_guess}/iter={batch_num}_gradient_{iteration}.png')
#                 plot_single(x0.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{num_vec}_{initial_guess}/iter={batch_num}_inversion_{iteration}.png')
#                 plot_single(output.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{num_vec}_{initial_guess}/iter={batch_num}_inversion_{iteration}_output.png')
#             else:
#                 plt.savefig(f'inversion_result_{loss_type}_{initial_guess}/iter={batch_num}_gradient_{iteration}.png')
#                 plot_single(x0.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{initial_guess}/iter={batch_num}_inversion_{iteration}.png')
#                 plot_single(output.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{initial_guess}/iter={batch_num}_inversion_{iteration}_output.png')
            
#         print(f"Iteration {iteration}, Loss: {loss_total.item():.4e}", inversion_MSE.item(), ssim_value)
#         posterior_set.append(x0.clone().detach().cpu().numpy())

#     return posterior_set, losses, inversion_MSEs, regs, ssims, output.detach().cpu().squeeze(), metrics_per_minute



# ----------------------
# Least Squares Posterior Estimation (with per-iteration timing)
# ----------------------
def least_squares_posterior_estimation(model, input_data, true_data, learning_rate,
                                       batch_num, num_iterations=500, prior=None):
    if loss_type != "Devito":
        model.eval()
    mse_loss = torch.nn.MSELoss()

    x0 = input_data.clone().detach().requires_grad_(True).to(device)
    posterior_set = []
    optimizer = torch.optim.Adam([x0], lr=learning_rate)

    losses, inversion_MSEs, regs, ssims = [], [], [], []
    # record *every* iteration, with timestamp
    loss_data_iter = []

    start_time = time.time()
    i = cols[:, 0].long()
    j = cols[:, 1].long()
    if sub_sampling == True:
        mask = torch.zeros((128, 128), dtype=torch.bool)
        mask[i, j] = True
        subsampled_mask = torch.zeros_like(mask)
        subsampled_mask[::6, ::6] = True
        final_mask = mask & subsampled_mask
        i, j = final_mask.nonzero(as_tuple=True)
        count = final_mask.sum().item()
        print(f"Number of True values: {count}")


    for iteration in range(num_iterations):
        optimizer.zero_grad()
        if loss_type == "Devito":
            squeezed_x0 = x0.squeeze()
            squeezed_x0.retain_grad()
            output = model(squeezed_x0)
        else:
            output = model(x0)

        # extract and compute loss
        if loss_type == "Devito":
            extracted_output = output[i, j]
            extracted_target = true_data[:, :, i, j]
            # x0 = x0.reshape(1, 1, x0.shape[0], x0.shape[1])
        else:
            extracted_output = output[:, :, i, j]
            extracted_target = true_data[:, :, i, j]

        loss = mse_loss(extracted_output.squeeze(), extracted_target.squeeze())
        reg = gradient_penalty(x0)
        loss_total = loss + alpha * reg
        loss_total.backward()
        optimizer.step()

        # metrics (L2)
        diff = x0 - prior
        inversion_MSE = torch.norm(diff) / torch.norm(prior)
        input_numpy = x0.detach().cpu().squeeze().numpy()
        prior_numpy = prior.detach().cpu().squeeze().numpy()
        ssim_value = ssim(input_numpy.astype(np.float64),
                          prior_numpy.astype(np.float64),
                          data_range=float(input_numpy.max() - input_numpy.min()))

        # elapsed time
        now = time.time()
        elapsed = now - start_time

        # record
        loss_data_iter.append({
            "sample":        batch_num,
            "iteration":     iteration,
            "elapsed_s":     elapsed,
            "loss":          loss_total.item(),
            "inversion_MSE": inversion_MSE.item(),
            "regularization":reg.item(),
            "SSIM":          ssim_value
        })

        # if batch_num < 2 and iteration % 50 == 0 and loss_type != "Devito":
        if batch_num < 2 and iteration % 50 == 0:
            gradient = x0.grad.detach().cpu().squeeze()  # shape: [H, W] or similar
            plt.imshow(gradient.numpy(), cmap='viridis')
            plt.colorbar(label='Gradient Value', shrink=0.8)
            plt.title('Gradient w.r.t. Input x0')
            if loss_type == "JAC":
                plt.savefig(f'inversion_result_{loss_type}_{num_vec}_{initial_guess}/iter={batch_num}_gradient_{iteration}.png')
                plot_single(x0.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{num_vec}_{initial_guess}/iter={batch_num}_inversion_{iteration}.png')
                plot_single(output.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{num_vec}_{initial_guess}/iter={batch_num}_inversion_{iteration}_output.png')
            else:
                plt.savefig(f'inversion_result_{loss_type}_{initial_guess}/iter={batch_num}_gradient_{iteration}.png')
                plot_single(x0.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{initial_guess}/iter={batch_num}_inversion_{iteration}.png')
                plot_single(output.detach().cpu().squeeze(), f'inversion_result_{loss_type}_{initial_guess}/iter={batch_num}_inversion_{iteration}_output.png')
            
        print(f"Iteration {iteration}, Loss: {loss_total.item():.4e}", inversion_MSE.item(), ssim_value)

        # store for plotting later
        losses.append(loss_total.item())
        inversion_MSEs.append(inversion_MSE.item())
        regs.append(reg.item())
        ssims.append(ssim_value)
        posterior_set.append(x0.clone().detach().cpu().numpy())

    return posterior_set, losses, inversion_MSEs, regs, ssims, output.detach().cpu().squeeze(), loss_data_iter, i, j


# ----------------------
# Main Script for Inversion on Multiple Samples (batch_size=1)
# ----------------------
if __name__ == "__main__":
    # Set up device and random seed.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print(f"Using device: {device}")

    # Define simulation parameters.
    num_vec = 1
    loss_type = "JAC"  # or "JAC" "MSE" "Devito"
    kernel_size = 45 #(grf, fullobs)
    sigma = 10.0 # (grf, fullobs)
    alpha = 0.05 #(grf, fullobs) (noisy, fullobs)
    noise_std = 0.4
    initial_guess = "prior_mean" # "smooth", "noisy"
    sub_sampling = True

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



    # # Iterate over the dataloader (each batch is one sample due to batch_size=1).
    # for batch in dataloader:
    #     x = batch['x'].to(device)
    #     y = batch['y'].to(device)
    #     L = batch['L'].view(-1).to(device)
    #     d = int(x.shape[-1])
    #     cols = torch.tensor([ (idx.item() // d, idx.item() % d) for idx in L ], device=device)

    #     # Run inversion for the current sample.
    #     if initial_guess == "smooth":
    #         zero_X = apply_gaussian_smoothing(x, kernel_size, sigma) + 1e-3
    #     elif initial_guess == "noisy":
    #         zero_X = x + torch.randn_like(x) * noise_std
    #     elif initial_guess == "prior_mean":
    #         zero_X = prior_mean
    #     plot_single(zero_X.detach().cpu().squeeze(), f"zero_X_sample_{sample_counter}.png", "jet")
        
    #     if loss_type == "Devito":
    #         forcing_term = torch.zeros(zero_X.squeeze().shape)
    #         groundwater_model = GroundwaterModel(forcing_term.shape[0])
    #         model = lambda x: groundwater_model(x, forcing_term) #500000 #@TODO
    #     posterior_set, losses, inversion_MSEs, regs, ssims, pred, metrics_per_minute = least_squares_posterior_estimation(
    #         model, zero_X, y, learning_rate, batch_num=sample_counter, num_iterations=num_epoch, prior=x
    #     )
        
    #     # Plot the final inversion result.
    #     final_x0 = torch.tensor(posterior_set[-1]).detach()
    #     plot_inversion_result(zero_X, x, y, pred, final_x0, loss_type, sample_counter)
        
    #     # Evaluate final metrics for the current sample.
    #     final_ssim = ssim(
    #         final_x0.squeeze().cpu().numpy().astype(np.float64),
    #         x.squeeze().cpu().numpy().astype(np.float64),
    #         data_range=float(final_x0.max()-final_x0.min())
    #     )
    #     final_l2 = torch.norm(final_x0.squeeze() - x.squeeze()).cpu().numpy()
    #     print(f"Sample {sample_counter} - Final SSIM: {final_ssim:.4f}, Final L2: {final_l2}")
        
    #     # Append final metrics to lists.
    #     final_ssim_list.append(final_ssim)
    #     final_l2_list.append(final_l2)
        
    #     # Record the inversion losses and metrics for every iteration for this sample.
    #     for itr, (loss_val, mse_val, reg_val, ssim_val) in enumerate(zip(losses, inversion_MSEs, regs, ssims)):
    #         loss_data_all.append({
    #             "sample": sample_counter,
    #             "iteration": itr,
    #             "loss": loss_val,
    #             "inversion_MSE": mse_val,
    #             "regularization": reg_val,
    #             "SSIM": ssim_val
    #         })
    #         metrics_all.extend(metrics_per_minute)
        
    #     sample_counter += 1

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

        posterior_set, losses, inversion_MSEs, regs, ssims, pred, loss_data_iter, i_idx, j_idx = (
            least_squares_posterior_estimation(
                model, zero_X, y,
                learning_rate, batch_num=sample_counter,
                num_iterations=num_epoch, prior=x
            )
        )

        # Plot the final inversion result.
        final_x0 = torch.tensor(posterior_set[-1]).detach()
        plot_inversion_result(zero_X, x, y, pred, final_x0, loss_type, sample_counter, i_idx, j_idx)

        # collect this sample’s iteration‐by‐iteration records
        loss_data_all.extend(loss_data_iter)
        sample_counter += 1

    # save to single CSV
    df = pd.DataFrame(loss_data_all)

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

    if loss_type == "JAC":
        csv_file = f"loss_statistics_multiple_samples_{loss_type}_{num_vec}_{initial_guess}.csv"
        # df_min.to_csv(f'metrics_per_minute_{loss_type}_{num_vec}_{initial_guess}.csv', index=False)
    else:
        csv_file = f"loss_statistics_multiple_samples_{loss_type}_{initial_guess}.csv"
        # df_min.to_csv(f'metrics_per_minute_{loss_type}_{initial_guess}.csv', index=False)
    df.to_csv(csv_file, index=False)
    print(f"Loss data saved to {csv_file}")