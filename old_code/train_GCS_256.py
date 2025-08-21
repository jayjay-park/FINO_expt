import torch
import torch.nn as nn
import torch.autograd.functional as F
import torch.optim as optim
import datetime
import time
import numpy as np
import argparse
import logging
import os
import csv
import h5py
import pandas as pd
from torch.func import vmap, vjp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sGCS
from functorch import vjp, vmap
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import SymLogNorm

from torch.utils.data import Dataset, DataLoader, TensorDataset
from physicsnemo.models.fno import FNO
from modulus.models.mlp.fully_connected import FullyConnected


### Dataset ###
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

### Compute Metric ###
# def plot_results_two(true1, pred1, path, true_name='True Saturation', pred_name='Predicted Saturation'):
#     plt.figure(figsize=(10, 5))
#     plt.rcParams.update({'font.size': 13})

#     plt.subplot(1, 2, 1)
#     t1 = true1.squeeze().cpu().numpy()
#     t1_range = max(abs(t1.min()), abs(t1.max()))
#     plt.imshow(t1, cmap='seismic', vmin=-t1_range, vmax=t1_range)
#     # plt.imshow(true1.cpu().numpy(), cmap='seismic', vmin=0.0, vmax=1.0)
#     plt.colorbar(fraction=0.045, pad=0.06)
#     plt.title(true_name)

#     plt.subplot(1, 2, 2)
#     p1 = pred1.squeeze().cpu().numpy()
#     p1_range = max(abs(p1.min()), abs(p1.max()))
#     plt.imshow(p1, cmap='seismic', vmin=-p1_range, vmax=p1_range)
#     # plt.imshow(pred1.cpu().numpy(), cmap='seismic', vmin=0.0, vmax=1.0)
#     plt.colorbar(fraction=0.045, pad=0.06)
#     plt.title(pred_name)

#     plt.savefig(path, dpi=150, bbox_inches='tight')
#     plt.close()

def plot_results_two(true1, pred1, path, true_name='True Saturation', pred_name='Predicted Saturation'):
    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 13})

    # Plot the true saturation
    plt.subplot(1, 2, 1)
    t1 = true1.squeeze().cpu().numpy()
    t1_range = max(abs(t1.min()), abs(t1.max()))*0.8
    norm_t1 = SymLogNorm(linthresh=0.1 * t1_range, vmin=-t1_range, vmax=t1_range)
    plt.imshow(t1, cmap='seismic', norm=norm_t1)
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title(true_name)

    # Plot the predicted saturation
    plt.subplot(1, 2, 2)
    p1 = pred1.squeeze().cpu().numpy()
    p1_range = max(abs(p1.min()), abs(p1.max()))*0.8
    norm_p1 = SymLogNorm(linthresh=0.1 * p1_range, vmin=-p1_range, vmax=p1_range)
    plt.imshow(p1, cmap='seismic', norm=norm_p1)
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title(pred_name)

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_results(true1, pred1, path, true_name='True Saturation', pred_name='Predicted Saturation', cmap_name='jet', seismic=False):
    plt.figure(figsize=(19, 5))
    plt.rcParams.update({'font.size': 16})

    # Plot the true data
    plt.subplot(1, 3, 1)
    if seismic:
        t1 = true1.squeeze().cpu().numpy()
        t1_range = max(abs(t1.min()), abs(t1.max()))
        linthresh = 0.1 * t1_range  # Linear threshold for symmetric log scaling
        plt.imshow(t1, cmap=cmap_name, norm=SymLogNorm(linthresh=linthresh, vmin=-t1_range, vmax=t1_range))
    else:
        plt.imshow(true1.squeeze().cpu().numpy(), cmap=cmap_name, vmin=0.0, vmax=1.0)
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title(true_name)

    # Plot the predicted data
    plt.subplot(1, 3, 2)
    if seismic:
        p1 = pred1.squeeze().cpu().numpy()
        p1_range = max(abs(p1.min()), abs(p1.max()))
        linthresh = 0.1 * p1_range  # Linear threshold for symmetric log scaling
        plt.imshow(p1, cmap=cmap_name, norm=SymLogNorm(linthresh=linthresh, vmin=-p1_range, vmax=p1_range))
    else:
        plt.imshow(pred1.squeeze().cpu().numpy(), cmap=cmap_name, vmin=0.0, vmax=1.0)
    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title(pred_name)

    # Plot the error map (always linear, using absolute error)
    plt.subplot(1, 3, 3)
    error1 = true1.squeeze().cpu().numpy() - pred1.squeeze().cpu().numpy()
    error_abs = np.abs(error1)
    vmax_error = error_abs.max()

    if seismic:
        # Set a linear threshold as 10% of the maximum error
        linthresh_error = 0.1 * vmax_error  
        norm_error = SymLogNorm(linthresh=linthresh_error, vmin=0.0, vmax=vmax_error)
        plt.imshow(error_abs, cmap='magma', norm=norm_error)
    else:
        plt.imshow(error_abs, cmap='magma', vmin=0.0, vmax=vmax_error)

    plt.colorbar(fraction=0.045, pad=0.06)
    plt.title('Error')

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_single(true1, path, cmap='Blues'):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})

    # Create a centered normalization around 0
    norm = colors.CenteredNorm()

    # Apply the norm both to the image and the colorbar
    ax = plt.imshow(true1, cmap=cmap, norm=norm)
    plt.colorbar(ax, fraction=0.045, pad=0.06, norm=norm)

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_single_abs(true1, path, cmap='Blues'):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 16})

    # Apply the norm both to the image and the colorbar
    ax = plt.imshow(true1, cmap=cmap)
    plt.colorbar(ax, fraction=0.045, pad=0.06)

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_multiple(figures, path, cmap='Blues'):
    fig, axes = plt.subplots(1, 4, figsize=(10, 10))  # Create 2 rows and 4 columns
    plt.rcParams.update({'font.size': 10})

    for i, (true1, ax) in enumerate(zip(figures, axes.flat)):  # Flatten axes to loop through them
        norm = colors.CenteredNorm()
        im = ax.imshow(true1, cmap=cmap, norm=norm)
        ax.set_title(f'Time step {i+1}')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04, norm=norm)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_multiple_abs(figures, path, cmap='Blues'):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Create 2 rows and 4 columns
    plt.rcParams.update({'font.size': 16})

    for i, (true1, ax) in enumerate(zip(figures, axes.flat)):  # Flatten axes to loop through them
        im = ax.imshow(true1, cmap=cmap)
        ax.set_title(f'Time step {i+1}')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_multiple_abs_sat(figures, path, cmap='jet'):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))  # Create 2 rows and 4 columns
    plt.rcParams.update({'font.size': 16})

    for i, (true1, ax) in enumerate(zip(figures, axes.flat)):  # Flatten axes to loop through them
        im = ax.imshow(true1, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(f'Time step {i+1}')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

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

    path = f"../plot/Loss/Checkpoint/FNO_GCS_vec:{args.num_vec}_{loss_type}_{epoch}.png"
    epochs = np.arange(len(mse_diff))

    fig, ax = plt.subplots()
    ax.semilogy(epochs, mse_diff, "P-", lw=1.0, ms=4.0, color="coral", label="MSE (Train)")
    ax.semilogy(epochs, test_diff, "P-", lw=1.0, ms=4.0, color="blue", label="MSE (Test)")
    if args.loss_type == "JAC":
        ax.semilogy(epochs, jac_diff_list, "P-", lw=1.0, ms=4.0, color="slateblue", label=r"$\|J^Tv - \hat{J}^Tv\|$")
    ax.set_xlabel("Epochs", fontsize=24)
    ax.set_ylabel("Loss", fontsize=24)
    ax.legend()
    plt.savefig(path, dpi=150, bbox_inches="tight")

    return

### Train ###
def main(logger, args, loss_type, dataloader, test_dataloader, True_j, vec, rolling, test_x):
    
    # Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)
    torch.cuda.empty_cache()

    # Define FNO3D
    model = FNO(
        in_channels=1,
        out_channels=1,
        decoder_layer_size=64,
        num_fno_layers=3,
        num_fno_modes=[2, 20, 20],
        padding=3,
        dimension=3,
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

    # Create vjp_batch
    num_vec = 8
    True_j = torch.tensor(True_j).float()
    True_j = True_j.reshape(-1, dataloader.batch_size, args.num_timestep, num_vec, args.nx, args.ny)
    print("True J Before", True_j.shape, "After True J", True_j.shape) #([7, 100, 8, 64, 64]) -> ([idx, batchsize, 8, 64, 64])
    vec = torch.tensor(vec)
    vec_batch = vec.reshape(-1, dataloader.batch_size, args.num_timestep, num_vec, args.nx, args.ny)
    print("vec", vec_batch.shape) # [5, 2, 5, 8, 256, 256]
    vec_batch = vec_batch.cuda().float()


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
            X = X.unsqueeze(1)
            Y = Y.unsqueeze(1)

            print("Batch: ", idx)
            # 1. update vjp and eigenvector
            target_vjp = True_j[idx].cuda()
            target_vjp = target_vjp.unsqueeze(1)
            target_vjp = target_vjp.permute(0, 1, 3, 2, 4, 5) # [2, 1, 8, 5, 256, 256]
            cur_vec_batch = vec_batch[idx] # eigenvectors for vJp | [2, 5, 8, 256, 256]
            cur_vec_batch = cur_vec_batch.permute(0, 2, 1, 3, 4) # [2, 8, 5, 256, 256]
            
            # MSE 
            if args.loss_type == "MSE":
                output = model(X)
                loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
                if (epoch % 5 == 0) and (idx == 0):
                    plot_single(X[0, 0, 0].detach().cpu().numpy(), f"../plot/GCS_vec_{args.num_vec}/training/MSE/K0.png", "viridis")
                    plot_multiple_abs_sat(Y[0].squeeze().detach().cpu().numpy(), f"../plot/GCS_vec_{args.num_vec}/training/MSE/true_sat.png")
                    plot_multiple_abs_sat(output[0].squeeze().detach().cpu().numpy(), f"../plot/GCS_vec_{args.num_vec}/training/MSE/learned_sat_{epoch}.png")
                    plot_multiple_abs_sat(abs(output[0]-Y[0]).squeeze().detach().cpu().numpy(), f"../plot/GCS_vec_{args.num_vec}/training/MSE/diff_sat_{epoch}.png", "magma")
                if (epoch % 10 == 0) and (idx == 0):
                    output, vjp_func = torch.func.vjp(model, X) # pass only single permeability field, thus, size is [2, 1, 1, 256, 256]
                    output = output.permute(0, 2, 1, 3, 4) # both output and Y shape is [2, 1, 5, 256, 256]
                    loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
                    vjp_out = torch.empty((args.batch_size, num_vec, 1, args.num_timestep, args.nx, args.ny),
                        device=X.device, dtype=X.dtype)
                    
                    for e in range(num_vec):
                        # bases now has shape [batch_size, 1, time, 256,256]
                        print("Computing vjp: ", e)
                        bases = cur_vec_batch[:, e].unsqueeze(1)
                        vjp_out[:, e] = vjp_func(bases)[0]
                    target_vjp, vjp_out = target_vjp.squeeze(), vjp_out.squeeze()

                    for t in range(args.num_timestep):
                        plot_results(Y[0, 0, t].detach().cpu(),  
                                    output[0, t, 0].detach().cpu(),  
                                    f"../plot/GCS_vec_{args.num_vec}/training/MSE/sat_timestep_{t+1}_epoch_{epoch}.png")

                        plot_results(target_vjp[0, t].detach().cpu(), 
                                    vjp_out[0, t].detach().cpu(),
                                    f"../plot/GCS_vec_{args.num_vec}/training/MSE/vjp_timestep_{t+1}_epoch_{epoch}.png", r"True $v^TJ$", r"Predicted $v^TJ$", "seismic", seismic=True)

            else:
            # Regularization Term 
                # 2. compute MSE and Regularization term
                output, vjp_func = torch.func.vjp(model, X) # pass only single permeability field, thus, size is [2, 1, 1, 256, 256]
                output = output.permute(0, 2, 1, 3, 4) # both output and Y shape is [2, 1, 5, 256, 256]
                loss = criterion(output.squeeze(), Y.squeeze()) / torch.norm(Y)
                vjp_out = torch.empty((args.batch_size, num_vec, 1, args.num_timestep, args.nx, args.ny),
                       device=X.device, dtype=X.dtype)
                
                for e in range(num_vec):
                    # bases now has shape [batch_size, 1, time, 256,256]
                    print("Computing vjp: ", e)
                    bases = cur_vec_batch[:, e].unsqueeze(1)
                    vjp_out[:, e] = vjp_func(bases)[0]
                target_vjp, vjp_out = target_vjp.squeeze(), vjp_out.squeeze()
    
                if (epoch == 0) and (idx == 0):
                    # Save permeability field (input)
                    plot_single(X[0, 0, 0].detach().cpu().numpy(), 
                                f"../plot/GCS_vec_{args.num_vec}/training/JAC/K.png", "viridis")

                    # Save eigenvectors and true vjp results
                    for t in range(args.num_timestep):  
                        print("shape", cur_vec_batch.shape, target_vjp.shape)
                        plot_results_two(cur_vec_batch[0, 0, t].detach().cpu(), 
                                    target_vjp[0, t].detach().cpu(),
                                    f"../plot/GCS_vec_{args.num_vec}/training/JAC/True_eigen_vjp_timestep_{t+1}.png", r"$v$", r"$v^TJ$")

                if (epoch % 5 == 0) and (idx == 0):
                    # Loop through each timestep to plot separately
                    for t in range(args.num_timestep):
                        print(Y.shape, output.shape, target_vjp.shape, vjp_out.shape)
                        plot_results(Y[0, 0, t].detach().cpu(),  
                                    output[0, t, 0].detach().cpu(),  
                                    f"../plot/GCS_vec_{args.num_vec}/training/JAC/sat_timestep_{t+1}_epoch_{epoch}.png")

                        plot_results(target_vjp[0, t].detach().cpu(), 
                                    vjp_out[0, t].detach().cpu(),
                                    f"../plot/GCS_vec_{args.num_vec}/training/JAC/vjp_timestep_{t+1}_epoch_{epoch}.png", r"True $v^TJ$", r"Predicted $v^TJ$", "seismic", seismic=True)

                jac_diff = criterion(target_vjp, vjp_out) / (torch.norm(target_vjp) + 1e-8)
                jac_misfit += jac_diff.detach().cpu().numpy()
                loss += jac_diff * args.reg_param

            print("step")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            full_loss += loss.item()
            idx += 1
        
        # Save loss
        mse_diff.append(full_loss)
        if args.loss_type == "JAC":
            jac_diff_list.append(jac_misfit)
        # Save time
        end_time = time.time()  
        elapsed_time_train.append(end_time - start_time)
        
        # Validation
        model.eval()
        with torch.no_grad():
            for X_test, Y_test in test_dataloader:
                X_test = X_test.unsqueeze(1)
                Y_test = Y_test.unsqueeze(1)
                print("Test", X_test.shape, Y_test.shape)
                X_test, Y_test = X_test.cuda().float(), Y_test.cuda().float()
                output = model(X_test)
                test_loss = criterion(output.squeeze(), Y_test) / torch.norm(Y_test) # relative error
                full_test_loss += test_loss.item()
            test_diff.append(full_test_loss)
        model.train()

        print(f"Epoch: {epoch}, Train Loss: {full_loss:.6f}, JAC misfit: {jac_misfit}, Test Loss: {full_test_loss:.6f}")
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"../test_result/Checkpoint/GCS_vec_{args.num_vec}_{loss_type}_{args.nx}_{args.num_train}_{epoch}.pth")
            plot_loss_checkpoint(epoch, loss_type, mse_diff, test_diff, jac_diff_list)
        if full_test_loss < lowest_loss:
            print("saved lowest loss model")
            lowest_loss = full_test_loss
            torch.save(model.state_dict(), f"../test_result/GCS_vec_{args.num_vec}_{loss_type}_best.pth")
            # Save plot
            X_test, Y_test = next(iter(test_dataloader))
            X_test, Y_test = X_test.unsqueeze(1).cuda().float(), Y_test.unsqueeze(1).cuda().float()
            with torch.no_grad():
                Y_pred = model(X_test)
            plot_path = f"../plot/GCS_vec_{args.num_vec}/FNO_GCS_lowest_vec_{args.num_vec}_{loss_type}_True.png"
            plot_multiple_abs_sat(Y_test[0].squeeze().cpu(), plot_path)
            plot_multiple_abs_sat(Y_pred[0].squeeze().detach().cpu(), f"../plot/GCS_vec_{args.num_vec}/FNO_GCS_lowest_{loss_type}_pred.png")
            plot_multiple_abs_sat(abs(Y_pred[0]-Y_test[0]).squeeze().detach().cpu(), f"../plot/GCS_vec_{args.num_vec}/FNO_GCS_lowest_{loss_type}_diff.png", "magma")
                

        if full_loss < args.threshold:
            print("Stopping early as the loss is below the threshold.")
            break

    print("Finished Computing")

    # ----------------------
    # Save model & loss:
    # ----------------------
    torch.save(model.state_dict(), f"../test_result/GCS_vec_{args.num_vec}_{loss_type}_full.pth")
    # Save the elapsed times
    with open(f'../test_result/Time/GCS_{args.loss_type}_{args.nx}_{args.num_train}.csv', 'w', newline='') as csvfile:
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
            with open(f'../test_result/Loss/GCS_vec_{args.num_vec}_{name}_{args.loss_type}_{args.nx}_{args.num_train}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Loss'])
                writer.writerows(enumerate(data, 1))
    print("Losses saved to CSV files.")

    # ----------------------
    # Create Loss Plot:
    # ----------------------
    print("Create loss plot")
    mse_diff = np.asarray(mse_diff)
    jac_diff_list = np.asarray(jac_diff_list)
    test_diff = np.asarray(test_diff)
    path = f"../plot/Loss/GCS_vec_{args.num_vec}_{loss_type}.png"

    # Remove the first few epochs (e.g., the first 5 epochs)
    start_epoch = 30
    mse_diff = mse_diff[start_epoch:]
    test_diff = test_diff[start_epoch:]
    jac_diff_list = jac_diff_list[start_epoch:]  # Only if JAC is relevant
    epochs = np.arange(len(mse_diff))
    fig, ax = plt.subplots()
    ax.semilogy(epochs, mse_diff, "P-", lw=1.0, ms=4.0, color="coral", label="MSE (Train)")
    ax.semilogy(epochs, test_diff, "P-", lw=1.0, ms=4.0, color="blue", label="MSE (Test)")

    if args.loss_type == "JAC":
        ax.semilogy(epochs, jac_diff_list, "P-", lw=1.0, ms=4.0, color="slateblue", label=r"$\|J^Tv - \hat{J}^Tv\|$")

    ax.set_xlabel("Epochs", fontsize=24)
    ax.set_ylabel("Loss", fontsize=24)
    ax.legend()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print("Plot saved.")


    logger.info("%s: %s", "Loss Type", str(loss_type))
    logger.info("%s: %s", "Batch Size", str(args.batch_size))
    logger.info("%s: %s", "Training Loss", str(full_loss/len(dataloader)))
    logger.info("%s: %s", "Final Test Loss", str(full_test_loss/len(test_dataloader)))
    logger.info("%s: %s", "Lowest Test Loss", str(lowest_loss))
    logger.info("%s: %s", "MSE diff", str(mse_diff[-1]))

    return model




if __name__ == "__main__":
    # Set device
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    print("device: ", device)

    # Set arguments (hyperparameters)
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epoch", type=int, default=500)
    parser.add_argument("--num_train", type=int, default=100)
    parser.add_argument("--num_test", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--loss_type", default="JAC", choices=["MSE", "JAC", "Sobolev", "Dissipative"])
    parser.add_argument("--reg_param", type=float, default=50.0) # 0.1 -> 2
    # Set arguments (dataset)
    parser.add_argument("--num_vec", type=int, default=88)
    parser.add_argument("--num_timestep", type=int, default=5)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=256)

    args = parser.parse_args()

    # Am I supposed to transpose? am I doing v^TJ - J^Tv..?


    # ------------------------
    # Set the initial setting:
    # ------------------------
    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    out_file = os.path.join("../test_result/", f"GCS_{start_time}.txt")
    logging.basicConfig(filename=out_file, filemode="w", level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)

    cmap = LinearSegmentedColormap.from_list(
        "cmap_name",
        ["#0000FF", "white", "#FF0000"]
    )

    set_y, set_vjp, set_eig, set_rolling = [], [], [], []
    num_sample = args.num_train + args.num_test

    # ----------------------
    # Load the K:
    # ----------------------
    with h5py.File("../data_generation/src/rescaled_200_fields.h5", "r") as f:
        K = f["K_subset"][:]  # This is a NumPy array
        print("Original shape of K:", K.shape)  # Should print (256, 512, 200)

    K_selected = np.concatenate((K[:, :, :40], K[:, :, 81:]), axis=2)
    print("Selected shape of K:", K_selected.shape)

    # ------------------------------------------
    # Transpose K, log transform, and normalize:
    # ------------------------------------------
    K_transposed = np.transpose(K, (2, 0, 1))
    K_min = K_transposed.min()
    K_max = K_transposed.max()
    set_x = (K_transposed - K_min) / (K_max - K_min)

    plt.imshow(set_x[0], cmap="viridis")
    plt.colorbar()
    plt.savefig("Downsampled_py.png", dpi=150, bbox_inches='tight')
    plt.close()


    # ----------------------
    # Load Everything:
    # ----------------------
    for s_idx in range(1, 201):
        if 40 < s_idx < 82:
            continue  # Skip samples 41 to 81
        print("idx", s_idx)
        if s_idx <= 40:
            states_path    = f'../data_generation/src/num_ev_8_stateonly/states_sample_{s_idx}.jld2'
        else:
            states_path = f'../data_generation/src/num_ev_8/saturation_sample_{s_idx}.jld2'

        with h5py.File(states_path, 'r') as f1, \
            h5py.File(f'../data_generation/src/num_ev_8/FIM_eigvec_sample_{s_idx}.jld2', 'r') as f2, \
            h5py.File(f'../data_generation/src/num_ev_8/FIM_vjp_sample_{s_idx}.jld2', 'r') as f3:

            if s_idx <= 40:
                states_refs = f1['single_stored_object'][:]  # Load the array of object references
            else:
                states_refs = f1['sat_series'][:]
            states_tensors = []

            # Loop over the references, dereference them, and convert to tensors
            for ref in states_refs:
                state_data = f1[ref][:]
                state_tensor = torch.tensor(state_data)
                states_tensors.append(state_tensor)
            
            eigvec = f2['single_stored_object'][:]
            vjp = f3['single_stored_object'][:]
            cur_vjp = torch.tensor(vjp).reshape(torch.tensor(vjp).shape[0], 8, args.nx, args.ny)[:, :args.num_vec]
            cur_eig = torch.tensor(eigvec).reshape(torch.tensor(eigvec).shape[0], 8, args.nx, args.ny)[:, :args.num_vec]

            set_y.append(torch.stack(states_tensors).reshape(args.num_timestep, args.nx, args.ny))
            set_vjp.append(cur_vjp[:args.num_timestep]) 
            set_eig.append(cur_eig[:args.num_timestep])

            if s_idx == 1:
                plot_single_abs(set_x[s_idx-1], f"../plot/GCS_vec_{args.num_vec}/K_{s_idx}.png", "viridis")
                plot_multiple_abs(set_y[s_idx-1], f"../plot/GCS_vec_{args.num_vec}/S_{s_idx}.png")
                if args.loss_type == "JAC":
                    plot_multiple(torch.tensor(set_eig[s_idx-1][0]), f"../plot/GCS_vec_{args.num_vec}/eig_{s_idx}_timestep_1.png", "seismic")

    print("set x", set_x.shape, len(set_y))
    train_x_raw = torch.tensor(set_x[:args.num_train])
    train_y_raw = torch.stack(set_y[:args.num_train])
    test_x_raw = torch.tensor(set_x[args.num_train:args.num_train+args.num_test])
    test_y_raw = torch.stack(set_y[args.num_train:args.num_train+args.num_test])

    train_x_raw = train_x_raw.unsqueeze(1)  # Now tensor is [25, 1, 64, 64]
    train_x_raw = train_x_raw.repeat(1, args.num_timestep, 1, 1)  # Now tensor is [25, 8, 64, 64]
    test_x_raw = test_x_raw.unsqueeze(1)  # Now tensor is [25, 1, 64, 64]
    test_x_raw = test_x_raw.repeat(1, args.num_timestep, 1, 1)  # Now tensor is [25, 8, 64, 64]

    train_vjp = torch.stack(set_vjp[:args.num_train]).reshape(-1, args.nx, args.ny)
    print("vjp norm", torch.norm(train_vjp), torch.max(train_vjp))
    train_vjp = train_vjp / torch.norm(train_vjp)

    set_eig = torch.stack(set_eig[:args.num_train])
    print("len: ", len(train_vjp), len(set_eig))
    print("len:", len(train_x_raw), len(train_y_raw), len(test_x_raw), len(test_y_raw))

    # Create subsets of the datasets
    train_dataset = CustomDataset(train_x_raw, train_y_raw)
    test_dataset = CustomDataset(test_x_raw, test_y_raw)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Mini-batch: ", len(train_loader), train_loader.batch_size)

    # train
    main(logger, args, args.loss_type, train_loader, test_loader, train_vjp, set_eig, None, test_x_raw)