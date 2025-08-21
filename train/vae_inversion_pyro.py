# vae_inversion_pyro.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from models.ns_inversion import NSModel
from utils import load_config, get_dataset
import os
from pathlib import Path
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------------------
# Encoder and Decoder Modules (standard PyTorch)
# -------------------------------------------
class Encoder(nn.Module):
    def __init__(self, y_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(y_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_logstd = nn.Linear(256, z_dim)

    def forward(self, y):
        h = self.net(y)
        mu = self.fc_mu(h)
        logstd = self.fc_logstd(h)
        return mu, torch.exp(logstd)

class Decoder(nn.Module):
    def __init__(self, z_dim, x_shape):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, np.prod(x_shape))
        )
        self.x_shape = x_shape

    def forward(self, z):
        return self.linear(z).view(-1, *self.x_shape)

# -------------------------------------------
# Pyro Model Definition
# -------------------------------------------
def make_vae_model(decoder, fno, sigma):
    def model(y_obs):
        batch_size = y_obs.shape[0]
        z_dim = decoder.linear[0].in_features
        with pyro.plate("batch", batch_size):
            z = pyro.sample("z", dist.Normal(0, 1).expand([z_dim]).to_event(1))
            x = decoder(z.to(next(decoder.parameters()).device))
            y_hat = fno(x)
            pyro.sample("obs", dist.Normal(y_hat, sigma).to_event(3), obs=y_obs)
    return model

# -------------------------------------------
# SVI Training Loop
# -------------------------------------------
def train_vae_with_pyro(fno, encoder, decoder, dataloader, config):
    pyro.clear_param_store()
    model = make_vae_model(decoder, fno, config['sigma'])

    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, pyro.optim.Adam({"lr": config['lr']}), loss=Trace_ELBO())

    loss_list = []
    for epoch in range(config['num_epochs']):
        total_loss = 0.0
        for batch in dataloader:
            y = batch['y'].to(config['device'])
            noise = torch.randn_like(y) * config['noise_std']
            y_noisy = y + noise
            loss = svi.step(y_noisy)
            total_loss += loss

        loss_list.append(total_loss)
        print(f"[Epoch {epoch}] ELBO: {-total_loss:.2f}")
    return guide, decoder, loss_list

# -------------------------------------------
# Entry Point with Configurable Loss Type
# -------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_config = {
        'loss_type': "MSE",
        'fno_ckpt': "checkpoints/DARCY_MSE/Darcy_training_epoch=249_val_rel_l2_loss=0.0009_MSE_May14.ckpt",
        'data_config_path': "output/n=400_e=200_m=FNO_s=RFS_l=JAC_20250615_133916/config.yaml",
        'z_dim': 64,
        'x_shape': (1, 128, 128),
        'lr': 1e-4,
        'num_epochs': 50,
        'sigma': 1.0,
        'noise_std': 0.01,
        'limit': 100,
        'device': device
    }

    if experiment_config['loss_type'] == "JAC":
        experiment_config['fno_ckpt'] = "checkpoints/.../JAC_model.ckpt"
        experiment_config['noise_std'] = 0.03
    elif experiment_config['loss_type'] == "Devito":
        raise NotImplementedError("Devito backend not supported in pyro setup")

    model_fno = NSModel.load_from_checkpoint(experiment_config['fno_ckpt']).eval().to(device)

    config = load_config(experiment_config['data_config_path'])
    dataset = get_dataset(config.experiment.dataset_type, config.data_settings)
    dataloader = dataset.get_dataloader(offset=0, limit=experiment_config['limit'])

    encoder = Encoder(y_dim=128 * 128, z_dim=experiment_config['z_dim']).to(device)
    decoder = Decoder(z_dim=experiment_config['z_dim'], x_shape=experiment_config['x_shape']).to(device)

    guide, decoder, loss_list = train_vae_with_pyro(model_fno, encoder, decoder, dataloader, experiment_config)

    save_dir = Path(f"vae_pyro_{experiment_config['loss_type']}")
    save_dir.mkdir(exist_ok=True)
    torch.save(guide.state_dict(), save_dir / "guide.pth")
    torch.save(decoder.state_dict(), save_dir / "decoder.pth")
    pd.DataFrame({"loss": loss_list}).to_csv(save_dir / "training_elbo.csv")
    plt.plot(loss_list)
    plt.title("SVI ELBO Loss")
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.savefig(save_dir / "elbo_curve.png")
    plt.close()

if __name__ == "__main__":
    main()
