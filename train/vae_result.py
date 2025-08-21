import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models.ns_inversion import NSModel
from train_vae import Encoder, Decoder  # assumes you saved these classes
from utils import load_config, get_dataset

def load_models(ckpt_path, z_dim, x_shape, device):
    fno = NSModel.load_from_checkpoint(ckpt_path).eval().to(device)

    encoder = Encoder(y_dim=128*128, z_dim=z_dim).to(device)
    decoder = Decoder(z_dim=z_dim, x_shape=x_shape).to(device)
    encoder.load_state_dict(torch.load("vae_ckpt/encoder.pth"))
    decoder.load_state_dict(torch.load("vae_ckpt/decoder.pth"))
    encoder.eval()
    decoder.eval()
    return fno, encoder, decoder

def run_posterior_sampling(fno, encoder, decoder, y_obs, num_samples=100):
    device = y_obs.device
    mu, std = encoder(y_obs)
    eps = torch.randn(num_samples, std.shape[-1], device=device)
    z_samples = mu + std * eps
    x_samples = decoder(z_samples)
    y_preds = fno(x_samples)

    return x_samples, y_preds

def plot_mean_std(x_samples, save_prefix="posterior"):
    x_samples_np = x_samples.detach().cpu().numpy()
    mean_x = x_samples_np.mean(axis=0).squeeze()
    std_x = x_samples_np.std(axis=0).squeeze()

    plt.imshow(mean_x, cmap='viridis')
    plt.colorbar()
    plt.title("Posterior Mean")
    plt.savefig(f"{save_prefix}_mean.png")
    plt.close()

    plt.imshow(std_x, cmap='inferno')
    plt.colorbar()
    plt.title("Posterior Std")
    plt.savefig(f"{save_prefix}_std.png")
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_dim = 64
    x_shape = (1, 128, 128)

    # Load models
    ckpt_path = "checkpoints/n=400_e=400_m=FNO_s=RFS_l=JAC_20250617_131205/n=400_e=400_m=FNO_s=RFS_l=JAC_epoch=299_val_rel_l2_loss=0.0156.ckpt"
    fno, encoder, decoder = load_models(ckpt_path, z_dim, x_shape, device)

    # Load one y_obs from dataset
    config = load_config("output/n=400_e=200_m=FNO_s=RFS_l=JAC_20250615_133916/config.yaml")
    dataset = get_dataset(config.experiment.dataset_type, config.data_settings)
    dataloader = dataset.get_dataloader(offset=0, limit=1)
    batch = next(iter(dataloader))
    y_obs = batch["y"].to(device)
    x = batch["x"]

    # Posterior samples
    x_samples, y_preds = run_posterior_sampling(fno, encoder, decoder, y_obs, num_samples=100)
    plot_mean_std(x_samples, save_prefix="posterior_inversion")
    plt.figure()
    plt.imshow(x.squeeze())
    plt.savefig("posterior_true")
