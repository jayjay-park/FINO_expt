import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from physicsnemo.models.fno import FNO
import h5py
# from torchmetrics.image import StructuralSimilarityIndexMeasure
from skimage.metrics import structural_similarity as ssim
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from matplotlib import colors
from scipy.interpolate import interp1d

import sys
sys.path.append('../test')
from generate_NS_org import *
from PINO_NS import *
from baseline import *

import torch.nn.functional as F

'''
Can compute average of but then inversion result will look a bit mo probabilistic.
'''

def gaussian_kernel(size: int, sigma: float):
    """Creates a 2D Gaussian kernel."""
    scale_factor = 1.

    # Create a 1D Gaussian kernel
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    
    # Create a 2D Gaussian kernel by outer product
    kernel = gauss[:, None] @ gauss[None, :]
    kernel *= scale_factor

    return kernel

def apply_gaussian_smoothing(batch_matrix: torch.Tensor, kernel_size: int, sigma: float):
    """Applies Gaussian smoothing to a batch of input matrices using a Gaussian kernel."""
    # Create the Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    
    # Reshape the kernel to be compatible with 2D convolution
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: 1 x 1 x kernel_size x kernel_size
    
    # Expand the kernel for each channel
    kernel = kernel.expand(1, 1, kernel_size, kernel_size).cuda()  # Shape: 8 x 1 x kernel_size x kernel_size

    # Compute the range of the input field
    original_min = batch_matrix.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    original_max = batch_matrix.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    original_range = original_max - original_min
    
    # Apply 2D convolution to smooth each matrix in the batch
    smoothed_batch = F.conv2d(batch_matrix, kernel, padding=kernel_size // 2, groups=1)

    
    # return smoothed_batch  # Output shape: batch_size x 8 x 64 x 64
    # Compute the range of the smoothed field
    smoothed_min = smoothed_batch.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    smoothed_max = smoothed_batch.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    smoothed_range = smoothed_max - smoothed_min
    
    # Rescale the smoothed field to the original range
    rescaled_batch = (smoothed_batch - smoothed_min) / (smoothed_range + 1e-8) * original_range + original_min
    
    return rescaled_batch  # Output shape: batch_size x 1 x 64 x 64

def plot_single(true1, path, cmap='Blues', vmin=None, vmax=None):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})
    
    # norm = colors.CenteredNorm()
    # Use a fixed range for vmin and vmax, if provided
    # print("vmin", vmin, vmax)
    if vmin != 0:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmin is not None and vmax is not None else colors.CenteredNorm()
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax) if vmin is not None and vmax is not None else colors.CenteredNorm()
    
    # Initialize ax properly and plot the image
    fig, ax = plt.subplots()
    cax = ax.imshow(true1, cmap=cmap, norm=norm)
    
    # Add colorbar
    plt.colorbar(cax, ax=ax, fraction=0.045, pad=0.06)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the plot to the specified path
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_single_abs(true1, path, cmap='Blues', vmin=None, vmax=None):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})

    # Use Normalize to set the colorbar range to vmin and vmax
    norm = colors.Normalize(vmin=vmin, vmax=vmax) if vmin is not None and vmax is not None else colors.Normalize()

    # Plot the image and get the AxesImage object
    im = plt.imshow(true1, cmap=cmap, norm=norm)
    
    # Get the current axis and add the colorbar
    cbar = plt.colorbar(im, fraction=0.045, pad=0.06)
    
    # Set ticks on the axis
    ax = plt.gca()  # Get the current Axes object
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks

    # Save the plot to the specified path
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_single(true1, path, cmap="jet", vmin=None, vmax=None):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})
    
    # norm = colors.CenteredNorm()
    # Use a fixed range for vmin and vmax, if provided
    print("vmin", vmin, vmax)
    if vmin != 0:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmin is not None and vmax is not None else colors.CenteredNorm()
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax) if vmin is not None and vmax is not None else colors.CenteredNorm()
    
    # Initialize ax properly and plot the image
    fig, ax = plt.subplots()
    cax = ax.imshow(true1, cmap=cmap, norm=norm)
    
    # Add colorbar
    plt.colorbar(cax, ax=ax, fraction=0.045, pad=0.06)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the plot to the specified path
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_diff_with_shared_colorbar_all(figures, t, ssim_mse, ssim_pbi, path, cmap='Blues'):
    fig, axes = plt.subplots(1, 6, figsize=(35, 5))  # 1 row, 6 columns
    plt.rcParams.update({'font.size': 13})

    # Plot the first two subplots (with individual color bars)
    for i, ax in enumerate(axes[:2]):
        if i == 0:
            # Display the first image with overlapping second image
            im1 = ax.imshow(figures[0], cmap="jet", alpha=0.9)  # First image
            # im2 = ax.imshow(figures[1], cmap='Reds', alpha=0.4)  # Second image
            ax.set_title('True K0')
            
            # Add colorbars for both images if needed
            fig.colorbar(im1, ax=ax, fraction=0.045, pad=0.06)
            # Optionally add a second colorbar for `im2` if necessary:
            # fig.colorbar(im2, ax=ax, fraction=0.045, pad=0.06)

        elif i == 1:
            # Display the second subplot image
            im = ax.imshow(figures[i], cmap="jet")
            ax.set_title('H(K0)')
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.06)
        ax.set_xticks([])
        ax.set_yticks([])

    # Normalize color scale for the third and fourth subplots
    norm = plt.Normalize(vmin=np.min(figures[2:4]), vmax=np.max(figures[2:4]))  # Normalize color bar for axes[2] and axes[3]
    norm_2 = plt.Normalize(vmin=0, vmax=np.max(figures[4:]))  # Normalize color bar

    # Plot the third and fourth subplots with shared color bar
    for i, ax in enumerate(axes[2:4], 2):
        im = ax.imshow(figures[i], cmap="jet", norm=norm)
        if i == 2:
            ax.set_title(f'FNO | SSIM={ssim_mse:.4f}')
        elif i == 3:
            ax.set_title(f'DeFINO | SSIM={ssim_pbi:.4f}')
            # Create a shared color bar for the third and fourth subplots
            fig.colorbar(im, ax=axes[2:4], fraction=0.02, pad=0.06)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot the last two subplots (with individual color bars)
    for i, ax in enumerate(axes[4:], 4):
        im = ax.imshow(figures[i], cmap='inferno', norm=norm_2)
        if i == 4:
            ax.set_title(f'Abs Diff FNO')
        elif i == 5:
            ax.set_title(f'Abs Diff DeFINO')
            fig.colorbar(im, ax=axes[4:], fraction=0.02, pad=0.06)
        ax.set_xticks([])
        ax.set_yticks([])

    # Save and close
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_multiple(figures, path, cmap='Blues'):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Create 2 rows and 4 columns
    plt.rcParams.update({'font.size': 16})

    for i, (true1, ax) in enumerate(zip(figures, axes.flat)):  # Flatten axes to loop through them
        norm = colors.CenteredNorm()
        im = ax.imshow(true1, cmap=cmap, norm=norm)
        ax.set_title(f'Time step {i+1}')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04, norm=norm)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_multiple_abs(figures, path, cmap='Blues', vmin=None, vmax=None, mse=None):
    # Select the 4th and the last figure from the figures list
    selected_figures = [figures[3], figures[-1]]

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))  # Create 2 rows and 1 column
    plt.rcParams.update({'font.size': 16})
    
    # Use Normalize to set the colorbar range to vmin and vmax
    norm = colors.Normalize(vmin=vmin, vmax=vmax) if vmin is not None and vmax is not None else colors.Normalize()

    for i, (true1, ax) in enumerate(zip(selected_figures, axes)):  # Loop through the selected figures
        if mse == True:
            true1 = abs(true1)
        im = ax.imshow(true1, cmap=cmap, norm=norm)
        ax.set_title(f'Time Step {(i+1)*4}')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_share_bar(data, path, cmap='magma'):

    # Set global font size
    plt.rcParams.update({'font.size': 12})  # Set desired font size globally

    # Create subplots
    fig, axs = plt.subplots(2, 8, figsize=(16, 4), sharex=True, sharey=True)
    vmax = max(torch.max(d) for d in data)

    # Plot each subplot
    for i, ax in enumerate(axs.flatten()):
        c = ax.imshow(data[i], vmin=0, vmax=vmax, cmap=cmap)
        if i < 8:
            ax.set_title(f'Time Step {i+1}')
            ax.set_xticks([])
            ax.set_yticks([])

    # Create a single colorbar
    cbar = fig.colorbar(c, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_dataset(num_samples, num_init, time_step, nx=50, ny=50):
    input, output, init = [], [], []

    L1, L2 = 2*math.pi, 2*math.pi  # Domain size
    Re = 1000  # Reynolds number
    # Define a forcing function (or set to None)
    t = torch.linspace(0, 1, nx + 1, device="cuda")
    t = t[0:-1]
    X, Y = torch.meshgrid(t, t)
    forcing = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

    # Initialize Navier-Stokes solver
    ns_solver = NavierStokes2d(nx, ny, L1=L1, L2=L2, device="cuda")
    num_iter = int(num_samples/num_init)
    print("num_init: ", num_init)
    print("time step: ", num_iter)
    for s in range(num_init):
        print("gen data for init: ", s)
        
        # Generate initial vorticity field
        random_seed=42 + s
        w = gaussian_random_field_2d((nx, ny), 20, random_seed)
        init.append(w)
        w_current = w.cuda()
        vorticity_data = [w_current.cpu().numpy()]

        # Solve the NS
        for i in range(num_iter):
            w_current = ns_solver(w_current, f=forcing, T=time_step, Re=Re)
            vorticity_data.append(w_current.cpu().numpy())
        
        input.append(vorticity_data[:-1])
        output.append(vorticity_data[1:])

    return input, output, init


# Set up the device and random seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
print(f"Using device: {device}")

# Define simulation parameters
L = 2 * math.pi  # domain size
set_x, set_y = [], []
batch_size = 1
MSE_output, JAC_output = [], []
ssim_value = 0.
num_vec = 10

num_col = 5
kernel_size = 25  # Example kernel size (should be odd)
sigma = 5.0  # Standard deviation of the Gaussian
learning_rate = 0.001 # 0.002 #previous one = 0.001
num_epoch = 301 #301, 1101
discretization = 128
alpha = 0.00001 # previous one = 0.00001

# Load MSE FNO
MSE_model = FNO(
        in_channels=1,  # Adjusted for vx and vy inputs
        out_channels=1, # Adjusted for wz output
        decoder_layer_size=128,
        num_fno_layers=6,
        num_fno_modes=[32,32],
        padding=3,
        dimension=2,
        latent_channels=64
    ).to(device)

# load JAC FNO
JAC_model = FNO(
        in_channels=1,  # Adjusted for vx and vy inputs
        out_channels=1, # Adjusted for wz output
        decoder_layer_size=128,
        num_fno_layers=6,
        num_fno_modes=[32,32],
        padding=3,
        dimension=2,
        latent_channels=64
    ).to(device)

# JAC_path = f"../test_result/best_model_FNO_NS_vort_JAC_vec={num_vec}.pth"
# JAC_path = f"../test_result/best_model_FNO_NS_vort_JAC_vec=40.pth"
JAC_path = f"../test_result/best_model_FNO_NS_vort_JAC_vec={num_vec}_dim={discretization}.pth"
JAC_model.load_state_dict(torch.load(JAC_path))
JAC_model.eval()

# MSE_path = f"../test_result/best_model_FNO_NS_vort_MSE_vec={num_vec}.pth" (num_vec = 20)
MSE_path = f"../test_result/best_model_FNO_NS_vort_MSE_vec={num_vec}_dim={discretization}.pth"
MSE_model.load_state_dict(torch.load(MSE_path))
MSE_model.eval()

num_datapoint=1000
num_init=100
num_iter = int(num_datapoint/num_init)
input, output, init = generate_dataset(num_datapoint, num_init, 0.05, discretization, discretization)

set_x = torch.stack(init).cuda().float()
output = torch.tensor(output)
set_y = output[:, -1].cuda().float()

print("dataset size", set_x.shape, set_y.shape)

plot_single(set_x[0].detach().cpu(), f"NS_sample/num_data={num_datapoint}_num_init={num_init}/init_0", cmap="jet")
plot_single(set_x[1].detach().cpu(), f"NS_sample/num_data={num_datapoint}_num_init={num_init}/init_1", cmap="jet")
plot_single(set_y[0].detach().cpu(), f"NS_sample/num_data={num_datapoint}_num_init={num_init}/obs_0", cmap="jet")
plot_single(set_y[1].detach().cpu(), f"NS_sample/num_data={num_datapoint}_num_init={num_init}/obs_1", cmap="jet")



num_batch = set_x.shape[0]
print("num_batch", num_batch, "batch size:", batch_size)

# Function for least squares posterior estimation, where the input_data is updated
def least_squares_posterior_estimation(model, input_data, true_data, model_type, learning_rate, batch_num, num_iterations=500, prior=None):
    model.eval()  # Ensure the model is in evaluation mode (weights won't be updated)
    mse_loss = torch.nn.MSELoss()  # MSE loss function

    # Set the input data as a tensor that requires gradients (so it can be optimized)
    input_data = input_data.clone().detach().squeeze().requires_grad_(True).to(device)
    posterior_set = []
    true_min = torch.min(prior) - 0.1
    true_max = torch.max(prior) + 0.1
    print("true min max", true_min + 0.1, true_max - 0.1)

    # Define an optimizer to update the input data (instead of the model parameters)
    optimizer = torch.optim.Adam([input_data], lr=learning_rate)
    losses, inversion_MSEs, regs, ssims = [], [], [], []
    min_mse, max_mse = [], []
    min_jac, max_jac = [], []

    for iteration in range(num_iterations):
        optimizer.zero_grad()  # Reset gradients
        first_input = input_data.unsqueeze(dim=0).unsqueeze(dim=1)
        for ts in range(num_iter):
            output = model(first_input)
            first_input = output
            # first_input.data = torch.clamp(first_input.data, min=true_min, max=true_max)
        output = output.squeeze()

        if batch_num < 2:
            plot_single(input_data.clone().detach().cpu().squeeze(), f'NS_sample/num_data={num_datapoint}_num_init={num_init}/iter={batch_num}_{model_type}_{iteration}')
        # mask is well operator here
        # print("mask", mask.shape, batch_num)
        # output = output * mask[batch_num].cuda().float().squeeze()
        # loss = mse_loss(output[:, :, 15:-15], true_data[:, :, 15:-15])
        # print("loss type:", model_type, torch.min(output), torch.max(output))
        if batch_num == 0:
            if model_type == "MSE":
                plot_single(output.detach().cpu().numpy(), f"NS_sample/num_data={num_datapoint}_num_init={num_init}/S/{model_type}_{batch_num}_{iteration}")
            else:
                plot_single(output.detach().cpu().numpy(), f"NS_sample/num_data={num_datapoint}_num_init={num_init}/S/{model_type}_{batch_num}_{iteration}")
            plot_single(true_data.squeeze().detach().cpu().numpy(), f"NS_sample/num_data={num_datapoint}_num_init={num_init}/S/True_{batch_num}")
        if model_type == "MSE":
            min_mse.append(torch.min(output).detach().cpu().numpy())
            max_mse.append(torch.max(output).detach().cpu().numpy())
        else:
            min_jac.append(torch.min(output).detach().cpu().numpy())
            max_jac.append(torch.max(output).detach().cpu().numpy())
        loss = mse_loss(output, true_data)
        reg = torch.norm(input_data)**2
        input_numpy = input_data.detach().cpu().squeeze().numpy()
        ssim_value = ssim(input_numpy, prior.detach().cpu().numpy(), data_range=input_numpy.max()-input_numpy.min())
        print("reg", reg)
        # losses.append(loss.item())
        regs.append(reg.item() * alpha)
        # regs.append(reg.item())
        ssims.append(ssim_value)
        loss += alpha * reg
        # loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()
        # input_data.data = torch.clamp(input_data.data, min=0, max=135)
        # input_data.data = torch.clamp(input_data.data, min=true_min, max=true_max)
        inversion_MSE = F.mse_loss(input_data, prior)
        inversion_MSEs.append(inversion_MSE.item())

        # Plot loss
        plt.figure(figsize=(10, 6))
        if model_type == "MSE":
            plt.plot(losses, label=f'FNO', color='red', marker='^')
            plt.plot(regs, label=f'FNO', color='orange', marker='^')
        else:
            plt.plot(losses, label='DeFINO', color='blue', marker='o')
            plt.plot(regs, label='DeFINO', color='green', marker='o')
        plt.xlabel('Iterations', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.legend(fontsize=15)
        plt.grid(True)
        plt.savefig(f'NS_sample/num_data={num_datapoint}_num_init={num_init}/loss_plot_{learning_rate}_{num_epoch}_{model_type}_{num_col}.png')
        plt.close()

        # if iteration % 50 == 0:
        print(f"Iteration {iteration}, {model_type} Loss: {loss.item()}")
        posterior_set.append(input_data.clone().detach().cpu().numpy())
        # plot_single(input_data.clone().detach().cpu(), f'NS_sample/num_data={num_datapoint}_num_init={num_init}/iter={num_batch}_{model_type}_{iteration}_afterclamp',)

    # Return the optimized input data (permeability K)
    return posterior_set, losses, inversion_MSEs, regs, ssims, min_mse, min_jac, max_mse, max_jac  # Detach from the computational graph

def gauss_newton_posterior_estimation(model, input_data, true_data, model_type, learning_rate, batch_num, num_iterations=500, prior=None):
    model.eval()  # Ensure the model is in evaluation mode (weights won't be updated)
    mse_loss = torch.nn.MSELoss()  # MSE loss function

    # Set the input data as a tensor that requires gradients (so it can be optimized)
    input_data = input_data.clone().detach().squeeze().requires_grad_(True).to(device)
    posterior_set = []
    true_min = torch.min(prior) - 0.1
    true_max = torch.max(prior) + 0.1

    losses, inversion_MSEs, regs, ssims = [], [], [], []

    for iteration in range(num_iterations):
        # Forward pass
        first_input = input_data.unsqueeze(dim=0).unsqueeze(dim=1)
        for ts in range(num_iter):
            output = model(first_input)
            first_input = output
        output = output.squeeze()

        # Compute residuals and loss
        residuals = (output - true_data).flatten()
        loss = mse_loss(output, true_data)
        losses.append(loss.item())

        # Compute Jacobian using vector-Jacobian product (vjp)
        J = []
        for i in range(len(residuals)):
            grad = torch.autograd.grad(residuals[i], input_data, retain_graph=True, create_graph=True)[0]
            J.append(grad.flatten())
        J = torch.stack(J)  # Jacobian matrix

        # Gauss-Newton update: Δx = - (JᵀJ)^(-1) Jᵀr
        JTJ = J.T @ J + 1e-6 * torch.eye(J.shape[1], device=device)  # Add small regularization term for stability
        JTr = J.T @ residuals
        delta_x = torch.linalg.solve(JTJ, -JTr)  # Solve the linear system

        # Update input_data
        input_data = input_data + delta_x.view_as(input_data)
        input_data.data = torch.clamp(input_data.data, min=true_min, max=true_max)  # Ensure constraints

        # Regularization and monitoring
        reg = torch.norm(input_data) ** 2
        regs.append(reg.item())
        inversion_MSE = F.mse_loss(input_data, prior)
        inversion_MSEs.append(inversion_MSE.item())

        print(f"Iteration {iteration}, Loss: {loss.item()}, Regularization: {reg.item()}")
        posterior_set.append(input_data.clone().detach().cpu().numpy())

        # Optional: Plotting results (as in your original code)

    # Return the optimized input data and monitoring metrics
    return posterior_set, losses, inversion_MSEs, regs










# MLE: Iterate over epochs first
for epoch in range(0, num_epoch, 10):
    print(f"Epoch {epoch}: Performing least squares posterior estimation")

    ssim_all_batch_mse, ssim_all_batch_jac = 0., 0.
    mse_jac, mse_mse = 0.0, 0.0

    # run it for one time!!!
    if epoch == 0:
        set_interpolate_X = []
        posterior_estimate_jac_all, posterior_estimate_mse_all = [], []
        mse_loss_all, jac_loss_all = [], []
        mse_inversion_loss_all, jac_inversion_loss_all = [], []
        mse_regs_all, jac_regs_all = [], []
        mse_ssims_all, jac_ssims_all = [], []
        loss_all = []
        for i in range(num_batch):
            print("batch", i)
            X = set_x[i].to(device).float()  # Input permeability [batch_size, 8, 64, 64]
            zero_X = apply_gaussian_smoothing(X.unsqueeze(0), kernel_size, sigma)
            set_interpolate_X.append(zero_X.clone().detach().cpu())
            print("zero X", zero_X.shape, X.shape)
            Y_true = set_y[i].to(device).float()  # True observation S [batch_size, 8, 64, 64]

            if i == 0:
                # plot_single_abs(interpolate_X.squeeze().detach().cpu()[0], f"mask_updated.png", "Blues")
                plot_single(zero_X.squeeze().detach().cpu(), f"zero_X.png", "jet")

            # Call MLE for each sample in the batch within the current epoch
            posterior_estimate_mse, mse_losses, mse_inversion_losses, mse_regs, mse_ssims, min_mse, min_jac, max_mse, max_jac = least_squares_posterior_estimation(
                MSE_model, zero_X, Y_true, "MSE", learning_rate, i, num_iterations=num_epoch, prior = X
            )
            posterior_estimate_jac, jac_losses, jac_inversion_losses, jac_regs, jac_ssims, min_mse, min_jac, max_mse, max_jac = least_squares_posterior_estimation(
                JAC_model, zero_X, Y_true, "JAC", learning_rate, i, num_iterations=num_epoch, prior=X
            )
            posterior_estimate_jac_all.append(posterior_estimate_jac)
            posterior_estimate_mse_all.append(posterior_estimate_mse)
            mse_loss_all.append(mse_losses)
            jac_loss_all.append(jac_losses)
            mse_inversion_loss_all.append(mse_inversion_losses)
            jac_inversion_loss_all.append(jac_inversion_losses)
            mse_regs_all.append(mse_regs)
            jac_regs_all.append(jac_regs)
            mse_ssims_all.append(mse_ssims)
            jac_ssims_all.append(jac_ssims)

        print("shape,", len(posterior_estimate_jac_all), len(posterior_estimate_mse_all), len(mse_loss_all), len(jac_loss_all))
        mse_loss_all = torch.tensor(mse_loss_all)
        jac_loss_all = torch.tensor(jac_loss_all)
        posterior_estimate_jac_all = torch.tensor(posterior_estimate_jac_all)
        posterior_estimate_mse_all = torch.tensor(posterior_estimate_mse_all)        
        mse_inversion_loss_all = torch.tensor(mse_inversion_loss_all)
        jac_inversion_loss_all = torch.tensor(jac_inversion_loss_all)
        mse_regs_all = torch.tensor(mse_regs_all)
        jac_regs_all = torch.tensor(jac_regs_all)
        mse_ssims_all = torch.tensor(mse_ssims_all)
        jac_ssims_all = torch.tensor(jac_ssims_all)

        print("loss shape", mse_loss_all.shape, jac_loss_all.shape) #torch.Size([num_batch, num_epoch]) torch.Size([2, 3])
        print(posterior_estimate_mse_all.shape, posterior_estimate_jac_all.shape) #torch.Size([2, 3, 5, 8, 64, 64]) [num_batch, num_epoch, batch_size, 8, 64, 64]
        # mse_loss_all = torch.sum(mse_loss_all, dim=0)
        # jac_loss_all = torch.sum(jac_loss_all, dim=0)
        # posterior_estimate_mse_all = posterior_estimate_mse_all.permute(1, 0, 2, 3, 4, 5)
        # posterior_estimate_jac_all = posterior_estimate_jac_all.permute(1, 0, 2, 3, 4, 5)
        mse_mean_loss = torch.mean(mse_loss_all, dim=0).cpu().numpy()  # Mean across batches for each epoch
        mse_std_loss = torch.std(mse_loss_all, dim=0).cpu().numpy()
        jac_mean_loss = torch.mean(jac_loss_all, dim=0).cpu().numpy()  # Mean across batches for each epoch
        jac_std_loss = torch.std(jac_loss_all, dim=0).cpu().numpy()
        mse_inv_mean_loss = torch.mean(mse_inversion_loss_all, dim=0).cpu().numpy()  # Mean across batches for each epoch
        mse_inv_std_loss = torch.std(mse_inversion_loss_all, dim=0).cpu().numpy()
        jac_inv_mean_loss = torch.mean(jac_inversion_loss_all, dim=0).cpu().numpy()  # Mean across batches for each epoch
        jac_inv_std_loss = torch.std(jac_inversion_loss_all, dim=0).cpu().numpy()
        mse_regs_mean_loss = torch.mean(mse_regs_all, dim=0).cpu().numpy()
        mse_regs_std_loss = torch.std(mse_regs_all, dim=0).cpu().numpy()
        jac_regs_mean_loss = torch.mean(jac_regs_all, dim=0).cpu().numpy()
        jac_regs_std_loss = torch.std(jac_regs_all, dim=0).cpu().numpy()
        mse_ssims_mean_loss = torch.mean(1 - mse_ssims_all, dim=0).cpu().numpy()
        mse_ssims_std_loss = torch.std(1 - mse_ssims_all, dim=0).cpu().numpy()
        jac_ssims_mean_loss = torch.mean(1 - jac_ssims_all, dim=0).cpu().numpy()
        jac_ssims_std_loss = torch.std(1 - jac_ssims_all, dim=0).cpu().numpy()

        import pandas as pd

        # Create a dictionary to organize data
        loss_data = {
            "Loss Type": [
                "MSE Mean", "MSE Std", "Jacobian Mean", "Jacobian Std", 
                "MSE Inversion Mean", "MSE Inversion Std", 
                "Jacobian Inversion Mean", "Jacobian Inversion Std",
                "MSE Regularization Mean", "MSE Regularization Std",
                "Jacobian Regularization Mean", "Jacobian Regularization Std",
                "MSE SSIMs Mean", "MSE SSIMs Std",
                "Jacobian SSIMs Mean", "Jacobian SSIMs Std"
            ],
            "Value": [
                mse_mean_loss, mse_std_loss, jac_mean_loss, jac_std_loss,
                mse_inv_mean_loss, mse_inv_std_loss,
                jac_inv_mean_loss, jac_inv_std_loss,
                mse_regs_mean_loss, mse_regs_std_loss,
                jac_regs_mean_loss, jac_regs_std_loss,
                mse_ssims_mean_loss, mse_ssims_std_loss,
                jac_ssims_mean_loss, jac_ssims_std_loss
            ]
        }

        # Convert the dictionary into a DataFrame
        df = pd.DataFrame(loss_data)
        csv_file = f"NS_sample/num_data={num_datapoint}_num_init={num_init}/loss_statistics.csv"
        df.to_csv(csv_file, index=False)
        print(f"Loss data saved to {csv_file}")

        # Apply the 'tab20' colormap for lines
        colormap = plt.cm.Paired(np.linspace(0, 1, 12))  # Generate distinct colors
        FNO1 = 'red'#colormap[5]
        FNO2 = 'orange'#colormap[4]
        DeFINO1 = 'blue'#colormap[1]
        DeFINO2 = 'green'#colormap[0]

        # Plot loss for each batch per epoch
        plt.figure(figsize=(10, 6))
        epochs = np.arange(0, num_epoch)  # Epoch numbers

        plt.plot(epochs[::20], mse_mean_loss[::20], label="FNO: MSE Mean", color=FNO1, marker='^', markersize=4)
        plt.fill_between(epochs[::20], 
                    mse_mean_loss[::20] - mse_std_loss[::20], 
                    mse_mean_loss[::20] + mse_std_loss[::20], 
                    color=FNO1, alpha=0.3)

        plt.plot(epochs[::20], jac_mean_loss[::20], label="DeFINO: MSE Mean", color=DeFINO1, marker='o', markersize=4)
        plt.fill_between(epochs[::20], 
                    jac_mean_loss[::20] - jac_std_loss[::20], 
                    jac_mean_loss[::20] + jac_std_loss[::20], 
                    color=DeFINO1, alpha=0.3)

        plt.plot(epochs[::20], mse_regs_mean_loss[::20], label=r"FNO: Mean of \alpha \| \hat a\|^2_F", color=FNO2, marker='^', markersize=4)
        plt.fill_between(epochs[::20], 
                    mse_regs_mean_loss[::20] - mse_regs_std_loss[::20], 
                    mse_regs_mean_loss[::20] + mse_regs_std_loss[::20], 
                    color=FNO2, alpha=0.3)

        plt.plot(epochs[::20], jac_regs_mean_loss[::20], label=r"DeFINO: Mean of \alpha \| \hat a\|^2_F", color=DeFINO2, marker='o', markersize=4)
        plt.fill_between(epochs[::20], 
                    jac_regs_mean_loss[::20] - jac_regs_std_loss[::20], 
                    jac_regs_mean_loss[::20] + jac_regs_std_loss[::20], 
                    color=DeFINO2, alpha=0.3)
        
        # plt.xscale('log')  # Log scale for x-axis
        plt.yscale('log')  # Log scale for y-axis

        plt.xlabel('Iterations', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.legend(fontsize=15)
        plt.grid(True)
        plt.savefig(f'NS_sample/num_data={num_datapoint}_num_init={num_init}/loss_plot_{learning_rate}_all.png')
        plt.close()

        # Plot loss for each batch per epoch
        plt.figure(figsize=(10, 6))
        epochs = np.arange(0, num_epoch)  # Epoch numbers
        plt.plot(epochs[::20], mse_inv_mean_loss[::20], label="FNO: Mean Error in MSE", color=FNO1, marker='^', markersize=4)
        plt.fill_between(
            epochs[::20], 
            mse_inv_mean_loss[::20] - mse_inv_std_loss[::20], 
            mse_inv_mean_loss[::20] + mse_inv_std_loss[::20], 
            color=FNO1, alpha=0.3
        )

        plt.plot(epochs[::20], jac_inv_mean_loss[::20], label="DeFINO: Mean Error in MSE", color=DeFINO1, marker='o', markersize=4)
        plt.fill_between(
            epochs[::20], 
            jac_inv_mean_loss[::20] - jac_inv_std_loss[::20], 
            jac_inv_mean_loss[::20] + jac_inv_std_loss[::20], 
            color=DeFINO1, alpha=0.3
        )

        plt.plot(epochs[::20], mse_ssims_mean_loss[::20], label="FNO: Mean of 1 - SSIM", color=FNO2, marker='^', markersize=4)
        plt.fill_between(
            epochs[::20], 
            mse_ssims_mean_loss[::20] - mse_ssims_std_loss[::20], 
            mse_ssims_mean_loss[::20] + mse_ssims_std_loss[::20], 
            color="pink", alpha=0.3
        )

        plt.plot(epochs[::20], jac_ssims_mean_loss[::20], label="DeFINO: Mean of 1 - SSIM", color=DeFINO2, marker='o', markersize=4)
        plt.fill_between(
            epochs[::20], 
            jac_ssims_mean_loss[::20] - jac_ssims_std_loss[::20], 
            jac_ssims_mean_loss[::20] + jac_ssims_std_loss[::20], 
            color=DeFINO2, alpha=0.3
        )

        plt.xlabel('Iterations', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.legend(fontsize=15)
        plt.grid(True)
        plt.savefig(f'NS_sample/num_data={num_datapoint}_num_init={num_init}/loss_inv_plot_{learning_rate}_all.png')
        plt.close()



        set_interpolate_X = torch.stack(set_interpolate_X)
        set_interpolate_X = set_interpolate_X.reshape(num_batch, batch_size, 1, discretization, discretization)
        print("interpolate X", set_interpolate_X.shape)

    # Calculations for SSIM and MSE, for every 50th epoch
    # torch.stack(posterior_estimate_jac).shape: [1, 5, 8, 64, 64]
    ssim_all_mse, ssim_all_jac = 0.0, 0.0
    print("post", torch.tensor(posterior_estimate_jac_all).shape)
    for i in range(num_batch):
        X = set_x[i].to(device).float()
        for b in range(batch_size):
            print("b", b, set_interpolate_X.shape, X.shape)
            # mask
            # mask_S = (Y_true[b][-1] != 0).int().detach().cpu()
            # masked_X = X.detach().cpu()
            # # masked_X[mask_S == 0] = 20
            # masked_posterior_estimate_mse_all = posterior_estimate_mse_all[i][epoch].clone().detach().cpu().numpy()
            # # masked_posterior_estimate_mse_all[mask_S == 0] = 20
            # masked_mse = masked_posterior_estimate_mse_all

            # masked_posterior_estimate_jac_all = posterior_estimate_jac_all[i][epoch].clone().detach().cpu().numpy()
            # # masked_posterior_estimate_jac_all[mask_S == 0] = 20
            # masked_jac = masked_posterior_estimate_jac_all

            # [num_batch, num_epoch, batch_size, 8, 64, 64]
            pe_mse = posterior_estimate_mse_all[i][epoch].clone().detach().cpu().numpy()
            pe_jac = posterior_estimate_jac_all[i][epoch].clone().detach().cpu().numpy()
            true_x = X.detach().cpu()
            print("pe_mse", pe_mse.shape, true_x.shape)
            abs_diff_mse = abs(pe_mse - true_x.numpy())
            abs_diff_jac = abs(pe_jac - true_x.numpy())
            plot_single(pe_jac, f"debug_{b}.png", "magma")
            
            # metrics
            mse_mse += F.mse_loss(torch.tensor(pe_mse), true_x) # Compute MSE
            mse_jac += F.mse_loss(torch.tensor(pe_jac), true_x)
            
            # ssim_mse = ssim(pe_mse, masked_X.numpy(), data_range=pe_mse.max()-pe_mse.min())
            # ssim_jac = ssim(pe_jac, masked_X.numpy(), data_range=pe_jac.max()-pe_jac.min())
            ssim_mse = ssim(pe_mse, true_x.numpy(), data_range=pe_mse.max()-pe_mse.min())
            ssim_jac = ssim(pe_jac, true_x.numpy(), data_range=pe_jac.max()-pe_jac.min())
            
            path = f'NS_sample/num_data={num_datapoint}_num_init={num_init}/posterior_{num_col}_{epoch}_{i}_{b}'
            plot_diff_with_shared_colorbar_all(
                [true_x, set_interpolate_X[i].squeeze().detach().cpu(), pe_mse, pe_jac, abs_diff_mse, abs_diff_jac], 
                epoch, ssim_mse, ssim_jac, path, cmap='magma'
            )
            # ssim_all_mse += ssim_mse
            # ssim_all_jac += ssim_jac
            # Save or visualize posterior results as needed
            print(f"Posterior estimate for epoch {epoch}, batch {i} completed.")
        ssim_all_batch_mse += ssim_mse
        ssim_all_batch_jac += ssim_jac
        print("MSE", ssim_mse, "JAC", ssim_jac, "All MSE", ssim_all_batch_mse, "All JAC", ssim_all_batch_jac)

    # Print SSIM metrics after each epoch
    # print(f"number of samples:", end_idx-start_idx)
    print(f"Epoch {epoch} - DeFINO SSIM per batch: {ssim_all_jac}, MSE SSIM per batch: {ssim_all_mse}")
    print(f"Epoch {epoch} - DeFINO SSIM Full: {ssim_all_batch_jac}, MSE SSIM Full: {ssim_all_batch_mse}")
    print(f"Epoch {epoch} - DeFINO MSE Full: {mse_jac}, MSE MSE Full: {mse_mse}")
    print(f"Epoch {epoch} - DeFINO Forward Loss: {jac_losses[-1]}, MSE Forward Loss: {mse_losses[-1]}")
    
# Previous DeFINO vec=20
# Epoch 300 - DeFINO SSIM per batch: 0.0, MSE SSIM per batch: 0.0       
# Epoch 300 - DeFINO SSIM Full: 0.9879456189291962, MSE SSIM Full: 1.2687226984456916
# Epoch 300 - DeFINO MSE Full: 1.8145394325256348, MSE MSE Full: 1.41(py311_env)

# DeFINO vec=40 w/o clamping
# Epoch 400 - DeFINO SSIM Full: 1.047813615644191, MSE SSIM Full: 1.19981827038948
# Epoch 400 - DeFINO MSE Full: 2.0751659870147705, MSE MSE Full: 1.5564117431640625
# Epoch 400 - DeFINO Forward Loss: 0.44156622886657715, MSE Forward Loss: 0.3124856948852539

# DeFINO vec=40 w/ clamping lr=0.05
# Epoch 400 - DeFINO SSIM Full: 0.9039985816114974, MSE SSIM Full: 1.1755591693044098
# Epoch 400 - DeFINO MSE Full: 2.080564022064209, MSE MSE Full: 1.5374059677124023
# Epoch 400 - DeFINO Forward Loss: 0.01954670622944832, MSE Forward Loss: 0.31233468651771545

# DeFINO vec=20 w/ clamping lr=0.01
# Epoch 400 - DeFINO SSIM Full: 1.0213981791692397, MSE SSIM Full: 1.5049639437461355
# Epoch 400 - DeFINO MSE Full: 1.6254589557647705, MSE MSE Full: 1.101181983947754
# Epoch 400 - DeFINO Forward Loss: 0.20501692593097687, MSE Forward Loss: 0.3185122311115265

# DeFINO vec=20 w/ clamping lr=0.03 kernel size=27
# Epoch 500 - DeFINO SSIM Full: 0.7941700999245923, MSE SSIM Full: 0.7095695516199246
# Epoch 500 - DeFINO MSE Full: 2.105470657348633, MSE MSE Full: 2.051607847213745
# Epoch 500 - DeFINO Forward Loss: 0.022038420662283897, MSE Forward Loss: 0.27916160225868225

# kernel size = 31, epoch = 400
# Epoch 400 - DeFINO SSIM Full: 0.482791694408688, MSE SSIM Full: 0.40717164055485544
# Epoch 400 - DeFINO MSE Full: 2.643160820007324, MSE MSE Full: 2.781921863555908
# Epoch 400 - DeFINO Forward Loss: 0.03008931875228882, MSE Forward Loss: 0.047346390783786774

# kernel size = 25, epoch = 500
# Epoch 500 - DeFINO SSIM Full: 0.7217082257516402, MSE SSIM Full: 0.5831010668220347
# Epoch 500 - DeFINO MSE Full: 2.322382688522339, MSE MSE Full: 2.527787208557129
# Epoch 500 - DeFINO Forward Loss: 0.02916901186108589, MSE Forward Loss: 0.05656110867857933

# kernel size = 27, after rescaling the initial guess, after adding reg term
# Epoch 700 - DeFINO SSIM Full: 1.3531433009998501, MSE SSIM Full: 1.0852938032274777
# Epoch 700 - DeFINO MSE Full: 1.5228183269500732, MSE MSE Full: 1.7501953840255737
# Epoch 700 - DeFINO Forward Loss: 0.08762259781360626, MSE Forward Loss: 0.163499116897583

# Epoch 800 - DeFINO SSIM Full: 0.8068251159145556, MSE SSIM Full: 0.6860073023617927
# Epoch 800 - DeFINO MSE Full: 2.1912407875061035, MSE MSE Full: 2.3123834133148193
# Epoch 800 - DeFINO Forward Loss: 0.03180135414004326, MSE Forward Loss: 0.06748461723327637

# lr = 0.0005
# Epoch 800 - DeFINO SSIM Full: 1.829888612731859, MSE SSIM Full: 1.5660594194852928
# Epoch 800 - DeFINO MSE Full: 1.1530035734176636, MSE MSE Full: 1.4104435443878174
# Epoch 800 - DeFINO Forward Loss: 0.38636600971221924, MSE Forward Loss: 0.48755648732185364

# lr = 0.001 epoch = 1100
# Epoch 1100 - DeFINO SSIM Full: 2.6206099263574814, MSE SSIM Full: 1.803870345708521
# Epoch 1100 - DeFINO MSE Full: 1.1655023097991943, MSE MSE Full: 2.017420530319214
# Epoch 1100 - DeFINO Forward Loss: 0.06074655055999756, MSE Forward Loss: 0.20841236412525177





# DeFINO SSIM: 0.5241 FNO SSIM: 0.3608
# DeFINO RMSE: 1.0796 FNO RMSE: 1.4204
# DeFINO Forward MSE: 0.0607 FNO Forward MSE: 0.2084
