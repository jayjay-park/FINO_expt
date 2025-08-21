# run_all_vae_experiments.py
import torch
import itertools
from pathlib import Path
from vae_inversion_pyro import train_vae_with_pyro, Encoder, Decoder, make_vae_model
from models.ns_inversion import NSModel
from utils import load_config, get_dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# All experiment configurations to loop over
LOSS_TYPES = ["MSE", "JAC"]
NOISE_STD = [0.01, 0.03]
SIGMAS = [1.0]
LIMIT = 100
NUM_EPOCHS = 50
Z_DIM = 64
X_SHAPE = (1, 128, 128)

# Dataset and FNO config map
FNO_CHECKPOINTS = {
    "MSE": "checkpoints/DARCY_MSE/Darcy_training_epoch=249_val_rel_l2_loss=0.0009_MSE_May14.ckpt",
    "JAC": "checkpoints/n=400_e=50_m=FNO_s=RFS_l=JAC_20250624_120949/n=400_e=50_m=FNO_s=RFS_l=JAC_epoch=199_val_rel_l2_loss=0.0206.ckpt"
}

DATA_CONFIG_PATH = "output/n=400_e=200_m=FNO_s=RFS_l=JAC_20250615_133916/config.yaml"

def evaluate_posterior_samples(guide, decoder, fno, outdir, num_samples=100):
    z_dist = guide(None)
    if isinstance(z_dist, dict):
        mu = z_dist.get("loc")
        std = z_dist.get("scale")
    else:
        mu = z_dist.loc
        std = z_dist.scale

    eps = torch.randn(num_samples, mu.shape[-1], device=mu.device)
    z_samples = mu.unsqueeze(0) + eps * std.unsqueeze(0)

    z_samples = z_samples.to(next(decoder.parameters()).device)
    x_samples = decoder(z_samples)
    y_preds = fno(x_samples)

    x_mean = x_samples.mean(dim=0).squeeze().detach().cpu().numpy()
    x_std = x_samples.std(dim=0).squeeze().detach().cpu().numpy()

    plt.imshow(x_mean, cmap="viridis")
    plt.title("Posterior Mean (x)")
    plt.colorbar()
    plt.savefig(outdir / "posterior_mean_x.png")
    plt.close()

    plt.imshow(x_std, cmap="inferno")
    plt.title("Posterior Std (x)")
    plt.colorbar()
    plt.savefig(outdir / "posterior_std_x.png")
    plt.close()

    y_mean = y_preds.mean(dim=0).squeeze().detach().cpu().numpy()
    y_std = y_preds.std(dim=0).squeeze().detach().cpu().numpy()

    plt.imshow(y_mean, cmap="jet")
    plt.title("Posterior Mean (y)")
    plt.colorbar()
    plt.savefig(outdir / "posterior_mean_y.png")
    plt.close()

    plt.imshow(y_std, cmap="magma")
    plt.title("Posterior Std (y)")
    plt.colorbar()
    plt.savefig(outdir / "posterior_std_y.png")
    plt.close()

def run_experiment(loss_type, noise_std, sigma, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load FNO model
    fno_ckpt = FNO_CHECKPOINTS[loss_type]
    fno = NSModel.load_from_checkpoint(fno_ckpt).eval().to(device)

    # Load dataset
    config = load_config(DATA_CONFIG_PATH)
    dataset = get_dataset(config.experiment.dataset_type, config.data_settings)
    dataloader = dataset.get_dataloader(offset=0, limit=LIMIT)

    # Build encoder/decoder
    encoder = Encoder(y_dim=128 * 128, z_dim=Z_DIM).to(device)
    decoder = Decoder(z_dim=Z_DIM, x_shape=X_SHAPE).to(device)

    # Setup experiment config
    experiment_config = {
        'loss_type': loss_type,
        'z_dim': Z_DIM,
        'x_shape': X_SHAPE,
        'lr': lr,
        'num_epochs': NUM_EPOCHS,
        'sigma': sigma,
        'noise_std': noise_std,
        'limit': LIMIT,
        'device': device
    }

    guide, decoder, loss_list = train_vae_with_pyro(fno, encoder, decoder, dataloader, experiment_config)

    # Save results
    tag = f"{loss_type}_noise={noise_std}_sigma={sigma}"
    outdir = Path("vae_experiments") / tag
    outdir.mkdir(parents=True, exist_ok=True)

    torch.save(guide.state_dict(), outdir / "guide.pth")
    torch.save(decoder.state_dict(), outdir / "decoder.pth")
    pd.DataFrame({"elbo": loss_list}).to_csv(outdir / "elbo.csv")
    plt.plot(loss_list)
    plt.title(f"ELBO ({tag})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(outdir / "elbo_curve.png")
    plt.close()

    evaluate_posterior_samples(guide, decoder, fno, outdir)
    print(f"Finished: {tag}")

if __name__ == "__main__":
    for loss_type, noise_std, sigma in itertools.product(LOSS_TYPES, NOISE_STD, SIGMAS):
        run_experiment(loss_type, noise_std, sigma)
