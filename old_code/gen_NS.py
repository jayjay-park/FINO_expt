import torch
import torch.nn as nn
import torch.autograd.functional as F
import torch.optim as optim
import datetime
import time
import numpy as np
import argparse
import json
import logging
import os
import csv
import pandas as pd
import tqdm
import math
from torch.func import vmap, vjp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from functorch import vjp, vmap
from torch.utils.data import Subset
import matplotlib.colors as colors

import sys
sys.path.append('../test')
from generate_NS_org import *
from PINO_NS import *
from baseline import *

from torch.utils.data import Dataset, DataLoader, TensorDataset
from physicsnemo.models.fno import FNO
from modulus.models.mlp.fully_connected import FullyConnected

class Timer:
    def __init__(self):
        self.elapsed_times = []

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.elapsed_times.append(self.elapsed_time)
        return False


### Dataset ###
def save_trajectory_as_single_plot(data, save_file='trajectory.png', cols=6):
    num = data.shape[0]  # Number of frames
    rows = int(np.ceil(num / cols))  # Determine the number of rows needed
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()  # Flatten to easily iterate over

    for i in range(num):
        axes[i].imshow(data[i], cmap='gnuplot')
        axes[i].set_title(f"Time Step {i}")
        axes[i].axis('off')  # Turn off axis for cleaner look

    # Turn off any unused axes (if num is not a perfect multiple of cols)
    for i in range(num, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_file)  # Save the plot as one image file
    plt.close()  # Close the plot to avoid showing it
    return

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
        if (s == 0 or s == 1):
            save_trajectory_as_single_plot(torch.tensor(vorticity_data[:-1]), save_file=f'../plot/NS_plot/vec={args.num_vec}/num_obs={args.num_obs}/trajx_{s}.png', cols=6)
            save_trajectory_as_single_plot(torch.tensor(vorticity_data[1:]), save_file=f'../plot/NS_plot/vec={args.num_vec}/num_obs={args.num_obs}/trajy_{s}.png', cols=6)


    return input, output, init


class NSDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

### Auxiliary Function ###
def save_dataset_to_csv(dataset, prefix):
    k_data = []
    
    for k in dataset:
        k_data.append(k)
    
    k_df = pd.DataFrame(k_data)
    k_df.to_csv(f'{prefix}', index=False)
    print(f"Saved {prefix} dataset to CSV files")
    return

def load_dataset_from_csv(prefix, nx, ny):
    df = pd.read_csv(f'{prefix}')

    print(f'df Length: {len(df)}')
    print(f'df Shape: {df.shape}')
    
    data = [torch.tensor(row.values).reshape(nx, ny) for _, row in df.iterrows()]
    
    return data

# def log_likelihood(data, model_output, noise_std):
#     return (1/(2*noise_std**2))*torch.sum((data - model_output)**2)


def compute_fim_NS(simulator, input, T_data, noise_std, train_mean, nx, ny, forcing, time_step, Re, input_index, s, init_iter, num_observations):
    input = input.requires_grad_().cuda()
    fim = torch.zeros((nx*ny, nx*ny), device='cpu')

    # Pre-compute time_step
    current_ts = s - init_index * init_iter

    # Generate isotrophic gaussian noise
    noise = torch.randn(num_observations, nx, ny, device='cuda') * noise_std #+ train_mean

    # Define the log-likelihood function
    def log_likelihood(x, single_noise):
        # T_pred = simulator(x, f=forcing, T=time_step, Re=Re)
        for i in range(current_ts):
            next = ns_solver(x, f=forcing, T=time_step, Re=Re)
            x = next
        T_diff_obs = x + single_noise
        return 0.5 * torch.norm(T_data.cuda() - T_diff_obs) / (noise_std**2)

    # Compute Jacobian for all observations at once using vmap and jacrev
    f = lambda x, noise: log_likelihood(x, noise)
    single_diff = torch.vmap(torch.func.jacrev(f, argnums=0), in_dims=(None, 0))
    jacobians = single_diff(input, noise)

    # Compute FIM
    print("jacobians shape:", jacobians.shape) #[num_observations, 64, 64]
    flat_jacobians = jacobians.reshape(num_observations, -1)
    # print("flat_jacobians", flat_jacobians.shape) #[num_observations, 4096]
    fim = torch.matmul(flat_jacobians.T.cpu(), flat_jacobians.cpu())
    # print("fim shape:", fim.shape) [4096, 4096]

    # Plot at specific iterations if needed
    if num_observations >= 10:
        fim_cpu = fim.detach().cpu()
        plot_single(fim_cpu, f"../plot/NS_plot/vec={args.num_vec}/num_obs={num_observations}/fim_{input_index}_9_t={s}.png", "viridis")
        plot_single(fim_cpu[:100,:100], f"../plot/NS_plot/vec={args.num_vec}/num_obs={num_observations}/fim_sub_{input_index}_9_t={s}.png", "viridis")
        plot_single(fim_cpu[:,0].reshape(nx, ny), f"../plot/NS_plot/vec={args.num_vec}/num_obs={num_observations}/fim_sub_reshape_{input_index}_9_t={s}.png", "viridis")
    
    if num_observations >= 100:
        fim_cpu = fim.detach().cpu()
        plot_single(fim_cpu, f"../plot/NS_plot/vec={args.num_vec}/num_obs={num_observations}/fim_{input_index}_99_t={s}.png", "viridis")
        plot_single(fim_cpu[:100,:100], f"../plot/NS_plot/vec={args.num_vec}/num_obs={num_observations}/fim_sub_{input_index}_99_t={s}.png", "viridis")
        plot_single(fim_cpu[:,0].reshape(nx, ny), f"../plot/NS_plot/vec={args.num_vec}/num_obs={num_observations}/fim_sub_reshape_{input_index}_9_t={s}.png", "viridis")
    
    torch.cuda.empty_cache()

    return fim

# args.noise, train_mean, nx, ny, forcing, args.time_step, Re, init_index, s, init_iter, num_observations=args.num_obs
def compute_fim_NS_scalable(simulator, input, T_data, noise_std, nx, ny, forcing, time_step, Re, init_index, s, init_iter, num_observations):
    
    '''
    init_index: number of different input parameter
    init_iter: length of time series for each input parameter
    s: index of training dataset (single vorticity in the time series)
    '''
    
    input = input.requires_grad_().cuda()
    # fim = torch.zeros((nx*ny, nx*ny), device='cpu')

    # Pre-compute time_step
    current_ts = s - init_index * init_iter

    # Generate isotrophic gaussian noise
    noise = torch.randn(num_observations, nx, ny, device='cuda') * noise_std #+ train_mean

    # Define the log-likelihood function
    def log_likelihood(x, single_noise):
        for i in range(current_ts):
            next = ns_solver(x, f=forcing, T=time_step, Re=Re)
            x = next
        T_diff_obs = x + single_noise
        return 0.5 * torch.norm(T_data.cuda() - T_diff_obs) / (noise_std**2)

    # Compute Jacobian for all observations at once using vmap and jacrev
    f = lambda x, noise: log_likelihood(x, noise)
    single_diff = torch.vmap(torch.func.jacrev(f, argnums=0), in_dims=(None, 0))
    jacobians = single_diff(input, noise)

    # Compute FIM
    print("jacobians shape:", jacobians.shape) #[num_observations, 64, 64]
    jacobians = jacobians.reshape(-1, nx*ny).T #[4096, 50]
    U,S,VT = torch.linalg.svd(jacobians, full_matrices=False)

    torch.cuda.empty_cache()

    return U, S

def plot_loss_checkpoint(epoch, loss_type, mse_diff, test_diff, jac_diff_list=None):
    # Create loss plot
    print("Create loss plot")
    if epoch < 510:
        mse_diff = np.asarray(mse_diff)
        jac_diff_list = np.asarray(jac_diff_list)
        test_diff = np.asarray(test_diff)
    else: 
        start_epoch = 30
        mse_diff = mse_diff[start_epoch:]
        test_diff = test_diff[start_epoch:]
        jac_diff_list = jac_diff_list[start_epoch:]  # Only if JAC is relevant

    path = f"../plot/Loss/checkpoint/FNO_NS_vort_{loss_type}_{epoch}.png"
    epochs = np.arange(len(mse_diff))

    fig, ax = plt.subplots()
    ax.plot(epochs, mse_diff, "P-", lw=1.0, ms=4.0, color="red", label="MSE (Train)")
    ax.plot(epochs, test_diff, "P-", lw=1.0, ms=4.0, color="blue", label="MSE (Test)")
    if args.loss_type == "JAC":
        ax.plot(epochs, jac_diff_list, "P-", lw=1.0, color="black", ms=4.0, label=r"$\|J^Tv - \hat{J}^Tv\|$")
    ax.set_xlabel("Epochs",fontsize=24)
    ax.set_ylabel("Loss", fontsize=24)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    return

### Compute Metric ###
def plot_results(true1, true2, pred1, pred2, path):
    plt.figure(figsize=(15, 10))
    plt.rcParams.update({'font.size': 16})

    plt.subplot(2, 3, 1)
    plt.imshow(true1.cpu().numpy(), cmap='jet')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('True vorticity')

    plt.subplot(2, 3, 2)
    plt.imshow(pred1.cpu().numpy(), cmap='jet')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('Predicted vorticity')

    plt.subplot(2, 3, 3)
    error1 = true1.cpu().numpy() - pred1.cpu().numpy()
    vmin, vmax = 0.0, max(abs(error1.min()), abs(error1.max()))
    plt.imshow(np.abs(error1), cmap='inferno', vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('Error')

    plt.subplot(2, 3, 4)
    plt.imshow(true2.cpu().numpy(), cmap='jet')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('True Vorticity')

    plt.subplot(2, 3, 5)
    plt.imshow(pred2.cpu().numpy(), cmap='jet')
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('Predicted Vorticity')

    plt.subplot(2, 3, 6)
    error2 = true2.cpu().numpy() - pred2.cpu().numpy()
    vmin, vmax = 0.0, max(abs(error2.min()), abs(error2.max()))
    plt.imshow(np.abs(error2), cmap='inferno', vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('Error')

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()



### Train ###

def main(logger, args, loss_type, dataloader, test_dataloader, vec, simulator):
    # Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    model = FNO(
        in_channels=1,  # Adjusted for vx and vy inputs
        out_channels=1, # Adjusted for wz output
        decoder_layer_size=128,
        num_fno_layers=6,
        num_fno_modes=[32,32],
        padding=3,
        dimension=2,
        latent_channels=64
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1e-3)
    nx, ny = args.nx, args.ny
    for first_batch in test_dataloader:
        print(first_batch)
        break  # Stop after printing the first batch

    # Gradient-matching and training logic
    if args.loss_type == "Sobolev":
        Sobolev_Loss = HsLoss_2d()
    elif args.loss_type == "Dissipative":
        Sobolev_Loss = HsLoss_2d()
        # DISSIPATIVE REGULARIZATION PARAMETERS
        # below, the number before multiplication by S is the radius in the L2 norm of the function space
        S=args.nx
        radius = 156.25 * S # radius of inner ball
        scale_down = 0.5 # rate at which to linearly scale down inputs
        loss_weight = 0.01 * (S**2) # normalized by L2 norm in function space
        radii = (radius, (525 * S) + radius) # inner and outer radii, in L2 norm of function space
        sampling_fn = sample_uniform_spherical_shell #numsampled is batch size
        target_fn = linear_scale_dissipative_target
        dissloss = nn.MSELoss(reduction='mean')

        modes = 20
        width = 64

        in_dim = 1
        out_dim = 1
    elif args.loss_type == "JAC":
        csv_filename = f'../data/true_j_NS_{args.nx}_{args.num_train}_{args.num_obs}_{args.num_vec}_{args.noise}.csv'
        if os.path.exists(csv_filename):
            # Load True_j
            True_j_flat = pd.read_csv(csv_filename).values
            print("len", True_j_flat.shape, len(dataloader)*dataloader.batch_size*nx*ny)
            True_j = torch.tensor(True_j_flat)[:len(dataloader)*dataloader.batch_size*args.num_vec, :].reshape(len(dataloader), dataloader.batch_size, args.num_vec, nx, ny)
            print(f"Data loaded from {csv_filename}")
            vec = vec.reshape(len(dataloader), dataloader.batch_size, args.num_vec, args.nx, args.ny)
        else:
            True_j = torch.zeros(len(dataloader), dataloader.batch_size, args.num_vec, args.nx, args.ny)
            f = lambda x: simulator(x, f=forcing, T=args.time_step, Re=Re)
            vec = vec.reshape(len(dataloader), dataloader.batch_size, args.num_vec, args.nx, args.ny)
            for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
                for i in range(batch_data.shape[0]):  # Iterate over each sample in the batch
                    # single sample [nx, ny]
                    x = batch_data[i]
                    output, vjp_tru_func = torch.func.vjp(f, x.cuda())
                    print(batch_idx, i, vec[batch_idx, i].shape)
                    for vec_idx in range(args.num_vec):
                        vjp = vjp_tru_func(vec[batch_idx, i, vec_idx].cuda())[0].detach().cpu()
                        print("vjp", vjp.shape)
                        True_j[batch_idx, i, vec_idx] = vjp
                    if (i < 10) and (batch_idx == 0):
                        plot_single(vjp, f'../plot/NS_plot/vec={args.num_vec}/FIM/num_obs={args.num_obs}/vjp_{i}.png')

            # Save True_j to a CSV file
            True_j_flat = True_j.detach().reshape(-1, nx * ny)  # Flatten the last two dimensions
            pd.DataFrame(True_j_flat.numpy()).to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}")
        # Create vec_batch
        True_j = True_j.float()


        # Create vec_batch
        print("after reshape eigvec:", vec.shape, True_j.shape)
        vec_batch = vec.cuda().float()


    ### Training Loop ###
    elapsed_time_train, mse_diff, jac_diff_list, test_diff = [], [], [], []
    lowest_loss = float('inf')

    print("Beginning training")
    for epoch in range(args.num_epoch):
        start_time = time.time()
        full_loss, full_test_loss, jac_misfit = 0.0, 0.0, 0.0
        idx = 0
        
        for X, Y in dataloader:
            X, Y = X.cuda().float(), Y.cuda().float()
            
            # MSE 
            optimizer.zero_grad()
            if args.loss_type == "MSE":
                output = model(X.unsqueeze(dim=1))
                loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
            elif args.loss_type == "Sobolev":
                output = model(X.unsqueeze(dim=1))
                loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
                sob_loss = Sobolev_Loss(output.squeeze(), Y.squeeze())
                loss += sob_loss
            elif args.loss_type == "Dissipative":
                output = model(X.unsqueeze(dim=1))
                loss = Sobolev_Loss(output.squeeze(), Y.squeeze())
                x_diss = torch.tensor(sampling_fn(X.shape[0], radii, (S, S, 2)), dtype=torch.float).to(device)
                y_diss = torch.tensor(target_fn(x_diss, scale_down), dtype=torch.float).to(device)
                out_diss = model(x_diss.reshape(-1, 2, S, S)).reshape(-1, out_dim)
                diss_loss = (1/(S**2)) * loss_weight * dissloss(out_diss, y_diss.reshape(-1, out_dim)) # weighted by 1 / (S**2)
                loss += diss_loss
            else:
            # GM
                target = True_j[idx].cuda()
                cur_vec_batch = vec_batch[idx]
                print(cur_vec_batch.shape)

                output, vjp_func = torch.func.vjp(model, X.unsqueeze(dim=1))
                loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
                vjp_out_list = []
                for e in range(args.num_vec):
                    # print("e", e, cur_vec_batch.shape, cur_vec_batch[:, e].unsqueeze(0).shape) #torch.Size([5, 8, 10, 64, 64]) torch.Size([5, 1, 8, 64, 64])
                    vjp_out_onevec = vjp_func(cur_vec_batch[:, e].unsqueeze(1))[0] # -> learned vjp
                    vjp_out_list.append(vjp_out_onevec)
                    vjp_out = torch.stack(vjp_out_list, dim=1)
                    # print("vjp out shape", vec_batch.shape, vjp_out.shape)
                if idx == 0:
                    plot_single(vec_batch[idx][0][0].detach().cpu(), f'../plot/NS_plot/vjp_true.png')
                    plot_single(vjp_out[0, 0, 0].detach().cpu(), f'../plot/NS_plot/vjp_pred.png')
                    print("target", target.shape, "vjp_out", vjp_out.shape)
                jac_diff = criterion(target.squeeze(), vjp_out.squeeze())
                jac_misfit += jac_diff.detach().cpu().numpy()
                loss += jac_diff * args.reg_param

            loss.backward(retain_graph=True)
            optimizer.step()
            full_loss += loss.item()
            idx += 1
        
        # Save loss
        mse_diff.append(abs(full_loss - jac_misfit))
        if args.loss_type == "JAC":
            jac_diff_list.append(jac_misfit)
        # Save time
        end_time = time.time()  
        elapsed_time_train.append(end_time - start_time)
        
        # Validation
        model.eval()
        with torch.no_grad():
            for X_test, Y_test in test_dataloader:
                X_test, Y_test = X_test.cuda().float(), Y_test.cuda().float()
                output = model(X_test.unsqueeze(dim=1))
                test_loss = criterion(output.squeeze(), Y_test) / torch.norm(Y_test)
                full_test_loss += test_loss.item()
            test_diff.append(full_test_loss)
        model.train()
        
        print(f"Epoch: {epoch}, Train Loss: {full_loss:.6f}, JAC misfit: {jac_misfit}, Test Loss: {full_test_loss:.6f}")
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"../test_result/Checkpoint/FNO_NS_vort_{loss_type}_{args.nx}_{args.num_train}_{epoch}.pth")
            with torch.no_grad():
                Y_pred = model(first_batch[0].float().cuda().unsqueeze(dim=1))
            plot_loss_checkpoint(epoch, loss_type, mse_diff, test_diff, jac_diff_list)
            plot_path = f"../plot/NS_plot/checkpoint/FNO_NS_vort_{loss_type}_{epoch}_{args.nx}.png"
            plot_results(first_batch[1][0].squeeze().cpu(), first_batch[1][1].squeeze().cpu(), Y_pred[0].squeeze(), Y_pred[1].squeeze(), plot_path)
                
        if full_test_loss < lowest_loss:
            print("saved lowest loss model")
            lowest_loss = full_test_loss
            torch.save(model.state_dict(), f"../test_result/best_model_FNO_NS_vort_{loss_type}_vec={args.num_vec}_dim={args.nx}.pth")
            # Save plot
            with torch.no_grad():
                Y_pred = model(first_batch[0].float().cuda().unsqueeze(dim=1))
            plot_path = f"../plot/NS_plot/vec={args.num_vec}/FNO_NS_vort_lowest_{loss_type}_{args.nx}.png"
            plot_results(first_batch[1][0].squeeze().cpu(), first_batch[1][1].squeeze().cpu(), Y_pred[0].squeeze(), Y_pred[1].squeeze(), plot_path)
                
        if full_loss < args.threshold:
            print("Stopping early as the loss is below the threshold.")
            break

    print("Finished Computing")

    # Save the model
    torch.save(model.state_dict(), f"../test_result/best_model_FNO_NS_vort_full epoch_{loss_type}_vec={args.num_vec}_dim={args.nx}.pth")
    # Save the elapsed times
    with open(f'../test_result/Time/FNO_NS_vort_{args.loss_type}_{args.nx}_{args.num_train}_vec={args.num_vec}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Elapsed Time (seconds)'])
        for epoch, elapsed_time in enumerate(elapsed_time_train, 1):
            writer.writerow([epoch, elapsed_time])
    # Save the losses
    loss_data = [
        (mse_diff, 'mse_loss'),
        (jac_diff_list, 'jac_loss') if args.loss_type == "JAC" else (None, None),
        (test_diff, 'test_loss')
    ]
    for data, name in loss_data:
        if data:
            with open(f'../test_result/Losses/NS_vort_{name}_{args.loss_type}_{args.nx}_{args.num_train}_vec={args.num_vec}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Loss'])
                writer.writerows(enumerate(data, 1))
    print("Losses saved to CSV files.")

    # Create loss plot
    print("Create loss plot")
    plot_loss_checkpoint(epoch, loss_type, mse_diff, test_diff, jac_diff_list)
    print("Plot saved.")


    logger.info("%s: %s", "Loss Type", str(loss_type))
    logger.info("%s: %s", "Batch Size", str(args.batch_size))
    logger.info("%s: %s", "Training Loss", str(full_loss/len(dataloader)))
    logger.info("%s: %s", "Test Loss", str(full_test_loss/len(test_dataloader)))
    logger.info("%s: %s", "MSE diff", str(mse_diff[-1]))

    return model


def plot_single(true1, path, cmap='gnuplot'):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})

    # Create a centered normalization around 0
    norm = colors.CenteredNorm()

    # Apply the norm both to the image and the colorbar
    ax = plt.imshow(true1, cmap=cmap, norm=norm)
    plt.colorbar(ax, fraction=0.045, pad=0.06, norm=norm)

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Set device
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    print("device: ", device)

    # Set arguments: https://github.com/neuraloperator/neuraloperator/blob/main/config/navier_stokes_config.yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epoch", type=int, default=300)
    parser.add_argument("--num_train", type=int, default=300) #8000
    parser.add_argument("--num_test", type=int, default=50)
    parser.add_argument("--num_sample", type=int, default=300) #8000
    parser.add_argument("--num_init", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--loss_type", default="MSE", choices=["MSE", "JAC", "Sobolev", "Dissipative"])
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--reg_param", type=float, default=0.05)
    parser.add_argument("--nu", type=float, default=0.01) # Viscosity
    parser.add_argument("--time_step", type=float, default=0.05) # time step
    parser.add_argument("--num_vec", type=int, default=10)
    parser.add_argument("--num_obs", type=float, default=10) # time step
    parser.add_argument("--generate_data", type=bool, default=True) # time step

    args = parser.parse_args()

    # Save initial settings
    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"FNO_NS_vort_{start_time}.txt")
    logging.basicConfig(filename=out_file, filemode="w", level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)

    # Define Simulator
    L1, L2 = 2*math.pi, 2*math.pi  # Domain size
    Re = 1000  # Reynolds number
    # Define a forcing function (or set to None)
    t = torch.linspace(0, 1, args.nx + 1, device="cuda")
    t = t[0:-1]
    X, Y = torch.meshgrid(t, t)
    forcing = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))
    ns_solver = NavierStokes2d(args.nx, args.ny, L1=L1, L2=L2, device="cuda")

    # Generate Training/Test Data
    trainx_file = f'../data/NS_vort/train_x_{args.ny}_{args.num_train}_{args.num_init}_{args.num_obs}_{args.num_vec}.csv'
    trainy_file = f'../data/NS_vort/train_y_{args.ny}_{args.num_train}_{args.num_init}_{args.num_obs}_{args.num_vec}.csv'
    testx_file = f'../data/NS_vort/test_x_{args.ny}_{args.num_test}_{args.num_init}_{args.num_obs}_{args.num_vec}.csv'
    testy_file = f'../data/NS_vort/test_y_{args.ny}_{args.num_test}_{args.num_init}_{args.num_obs}_{args.num_vec}.csv'
    # if not os.path.exists(trainx_file):
    if args.generate_data == True:
        print("Creating Dataset")
        input, output, init = generate_dataset(args.num_train + args.num_test, args.num_init, args.time_step, args.nx, args.ny)
        input = torch.tensor(input).reshape(-1, args.nx*args.ny)
        output = torch.tensor(output).reshape(-1, args.nx*args.ny)
        print("data size", len(input), len(output))

        train_x = NSDataset(input[:args.num_train].numpy())
        train_y = NSDataset(output[:args.num_train].numpy())
        test_x = NSDataset(input[args.num_train:].numpy())
        test_y = NSDataset(output[args.num_train:].numpy())
        # Save datasets to CSV files
        save_dataset_to_csv(train_x, trainx_file)
        save_dataset_to_csv(train_y, trainy_file)
        save_dataset_to_csv(test_x, testx_file)
        save_dataset_to_csv(test_y, testy_file)

    
    print("Loading Dataset")
    sample = load_dataset_from_csv(trainx_file, args.nx, args.ny)
    print("sample", len(sample), sample[0].shape)
    train_x_raw = load_dataset_from_csv(trainx_file, args.nx, args.ny)
    train_y_raw = load_dataset_from_csv(trainy_file, args.nx, args.ny)
    test_x_raw = load_dataset_from_csv(testx_file, args.nx, args.ny)
    test_y_raw = load_dataset_from_csv(testy_file, args.nx, args.ny)

    def normalize_to_range(x, new_min=-1.0, new_max=1.0):
        """
        Normalize the tensor x to the range [new_min, new_max]
        """
        old_min = torch.min(x)
        old_max = torch.max(x)
        x_norm = (new_max - new_min) * (x - old_min) / (old_max - old_min) + new_min
        return x_norm

    # Normalize each sample
    plot_single(train_x_raw[0].reshape(args.nx, args.ny), f'../plot/NS_plot/vec={args.num_vec}/input.png')
    plot_single(train_x_raw[-1].reshape(args.nx, args.ny), f'../plot/NS_plot/vec={args.num_vec}/output.png')

    # Create subsets of the datasets
    train_dataset = CustomDataset(train_x_raw, train_y_raw)
    test_dataset = CustomDataset(test_x_raw, test_y_raw)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Mini-batch: ", len(train_loader), train_loader.batch_size)
    train_mean = torch.mean(torch.stack(train_x_raw))
    print("train_mean", train_mean.shape)

    # compute FIM eigenvector
    if args.loss_type == "JAC":
        csv_filename = f'../data/NS_vort/eigvec/largest_eigvec_NS_{args.nx}_{args.num_train}_{args.num_obs}_vec={args.num_vec}_{args.noise}_{args.num_init}.csv'
        if os.path.exists(csv_filename):
            print("Loading largest eigenvector")
            largest_eigenvector = pd.read_csv(csv_filename).values
            largest_eigenvector = torch.tensor(largest_eigenvector)
        else:
            largest_eigenvector = []
            nx, ny = args.nx, args.ny
            print("Reloaded train: ", train_x_raw[0].shape)
            # Compute FIM
            init_iter = int((args.num_train + args.num_test)/args.num_init)
            init_index = 0
            input_param = init[init_index]
            for s in range(args.num_train):
                print("train index:", s, "init_index", init_index)
                # if s == 0:
                #     print("s", train_x_raw[s])
                #     print("init", init[init_index])
                #     # save gradient 
                #     grad_vorticity_x = np.gradient(input_param, axis=0)
                #     grad_vorticity_y = np.gradient(input_param, axis=1)
                #     mag_vorticity = np.sqrt(grad_vorticity_x**2 + grad_vorticity_y**2)
                #     plot_single(mag_vorticity, f'../plot/NS_plot/vec={args.num_vec}/init_grad_{s}.png', "viridis")
                # should be changed to initial state.
                if (s % (init_iter) == 0) and (s != 0):
                    init_index += 1
                    input_param = init[init_index]
                    # save gradient 
                    grad_vorticity_x = np.gradient(input_param, axis=0)
                    grad_vorticity_y = np.gradient(input_param, axis=1)
                    mag_vorticity = np.sqrt(grad_vorticity_x**2 + grad_vorticity_y**2)
                    plot_single(mag_vorticity, f'../plot/NS_plot/vec={args.num_vec}/init_grad_{s}.png', "viridis")
                    print("s-1", train_x_raw[s-1])
                    print("s", train_x_raw[s])
                    print("s+1", train_x_raw[s+1])
                    print("init", init[init_index])
                    
                # give single output
                # fim = compute_fim_NS(ns_solver, input_param, train_y_raw[s], args.noise, train_mean, nx, ny, forcing, args.time_step, Re, init_index, s, init_iter, num_observations=args.num_obs).detach().cpu()
                print("input_param", input_param.shape, train_y_raw[s].shape)
                eigenvec, eigenvalues = compute_fim_NS_scalable(ns_solver, input_param, train_y_raw[s], args.noise, nx, ny, forcing, args.time_step, Re, init_index, s, init_iter, num_observations=args.num_obs)
                print("eigenvec", eigenvec.shape)
                # Analyze the FIM
                # eigenvalues, eigenvec = torch.linalg.eigh(fim.cuda())
                largest_eigenvector.append(eigenvec.detach().cpu())
                if s < 10:
                    print("eigval: ", eigenvalues)
                    plot_single(train_x_raw[s].detach().cpu(), f'../plot/NS_plot/vec={args.num_vec}/FIM/num_obs={args.num_obs}/state_{s}.png', "gnuplot")
                    # plot_single(eigenvec[0].detach().cpu().reshape(args.nx, args.ny), f'../plot/NS_plot/vec={args.num_vec}/FIM/num_obs={args.num_obs}/eigenvec0_{s}.png', "viridis")
                    # plot_single(eigenvec[1].detach().cpu().reshape(args.nx, args.ny), f'../plot/NS_plot/vec={args.num_vec}/FIM/num_obs={args.num_obs}/eigenvec1_{s}.png', "viridis")
                    # plot_single(eigenvec[2].detach().cpu().reshape(args.nx, args.ny), f'../plot/NS_plot/vec={args.num_vec}/FIM/num_obs={args.num_obs}/eigenvec2_{s}.png', "viridis")
                    # plot_single(eigenvalues.detach().cpu().reshape(args.nx, args.ny), f'../plot/NS_plot/vec={args.num_vec}/FIM/num_obs={args.num_obs}/eigenvalues_{s}.png', "viridis")
            largest_eigenvector = torch.stack(largest_eigenvector)
            flattened = largest_eigenvector.numpy().reshape(-1, largest_eigenvector.shape[-1])
            pd.DataFrame(flattened).to_csv(csv_filename, index=False)

            # pd.DataFrame(largest_eigenvector.numpy()).to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}")


        print("largest eigenvector shape: ", largest_eigenvector.shape)
        largest_eigenvector = largest_eigenvector.reshape(-1, args.nx, args.ny)
    else:
        largest_eigenvector = None
    for data in enumerate(train_loader):
        print("from loader", data)
    # train
    main(logger, args, args.loss_type, train_loader, test_loader, largest_eigenvector, ns_solver)
