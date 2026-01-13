import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import gc

# --- Fixed Dummy FNO Implementation ---
class SimpleFNO(nn.Module):
    def __init__(self, in_channels, out_channels, num_fno_modes, latent_channels):
        super().__init__()
        self.modes1 = num_fno_modes[0]
        self.modes2 = num_fno_modes[1]
        self.latent_channels = latent_channels
        
        self.fc0 = nn.Linear(in_channels, latent_channels)
        
        # Spectral weights: [in_channels, out_channels, mode1, mode2]
        self.weights1 = nn.Parameter(torch.view_as_complex(torch.randn(latent_channels, latent_channels, self.modes1, self.modes2, 2) * 0.02))
        self.weights2 = nn.Parameter(torch.view_as_complex(torch.randn(latent_channels, latent_channels, self.modes1, self.modes2, 2) * 0.02))
        
        self.fc1 = nn.Linear(latent_channels, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # 1. Lifting
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2) 

        # 2. Iterate through 3 layers to match your "num_fno_layers=3"
        for _ in range(3):
            res = x
            x_ft = torch.fft.rfft2(x)
            out_ft = torch.zeros_like(x_ft)
            
            # Use weights (simplified for dummy purposes)
            out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1
            )
            
            x = torch.fft.irfft2(out_ft, s=(res.size(-2), res.size(-1)))
            x = F.gelu(x + res) # Non-linearity + Skip connection

        # 3. Projection
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)

import torch.nn.functional as F

# --- NSModel Class (Your original logic) ---
class NSModel(pl.LightningModule):
    def __init__(self, in_channels=2, out_channels=1, num_fno_modes=[150, 100], latent_channels=64, reg_param=0.01, train_eigen_count=8):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.model = SimpleFNO(in_channels, out_channels, num_fno_modes, latent_channels)
        self.reg_param = reg_param
        self.train_eigen_count = train_eigen_count

    def forward(self, x):
        return self.model(x)
    
    def relative_l2_loss(self, true, pred):
        return torch.norm(true - pred) / (torch.norm(true) + 1e-8)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        x, y = batch['x'], batch['y']
        
        # Standard Forward & Backward
        output = self.forward(x)
        rel_l2 = self.relative_l2_loss(y, output)
        self.manual_backward(rel_l2)

        # Jacobian Regularization Loop
        K = self.train_eigen_count
        v = batch['v']
        true_Jvp = batch['Jvp']
        
        for k in range(K):
            v_dir = v[..., k] # Shape [B, 2, 512, 256]
            
            _, jval = torch.autograd.functional.jvp(
                self.forward, x, v_dir, create_graph=True
            )
            
            target = true_Jvp[..., k] # Shape [B, 1, 512, 256]
            loss_k = self.relative_l2_loss(target, jval)
            
            scaled = (self.reg_param * loss_k) / K
            self.manual_backward(scaled)
            
            del jval, loss_k, scaled

        opt.step()
        self.log("train_l2", rel_l2, prog_bar=True)
        return rel_l2

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# --- Dummy Dataset ---
class DummyNSDataset(Dataset):
    def __init__(self, num_samples=10, k_eig=8):
        self.num_samples = num_samples
        self.k_eig = k_eig

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Matching your reshape logic (2 input channels, 1 output channel)
        return {
            'x': torch.randn(2, 512, 256),
            'y': torch.randn(1, 512, 256),
            'v': torch.randn(2, 512, 256, self.k_eig),
            'Jvp': torch.randn(1, 512, 256, self.k_eig)
        }

# --- Execution ---
def run_memory_test():
    # Start with batch size 1 to find the baseline
    dataset = DummyNSDataset(num_samples=5, k_eig=8)
    dataloader = DataLoader(dataset, batch_size=1)

    model = NSModel(train_eigen_count=8)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=1,
        fast_dev_run=2, 
        precision=32 
    )

    print("Checking 40GB GPU Capacity...")
    trainer.fit(model, dataloader)
    print("Test passed! Memory is sufficient.")

if __name__ == "__main__":
    run_memory_test()