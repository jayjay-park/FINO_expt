import torch
import numpy as np
import matplotlib.pyplot as plt
from .base import Simulator  # Assuming base class defines interface
import os, sys
# add project root (one level up) to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from groundwater.groundwater.utils import GaussianRandomField, plot_fields
from groundwater.groundwater.devito_op import GroundwaterEquation


class DarcySimulator(Simulator):
    def __init__(self, size=256, T=1.0, dtype=torch.float32):
        super().__init__()
        self.size = size
        self.dtype = dtype
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GroundwaterEquation(size)
        self.T = T

    def sample(self):
        # Step 1: Sample from Gaussian Random Field
        # grf = GaussianRandomField(2, self.size, alpha=35, tau=0.001, sigma=10000.0) #--- prev version
        grf = GaussianRandomField(2, self.size, alpha=2, tau=3)
        u_samples = grf.sample(1)

        # # Sample random fields
        u_samples[u_samples>=0] = 0.9
        u_samples[u_samples<0] = 0.1
        
        return torch.tensor(u_samples[0], dtype=self.dtype, device=self.device)

    def forward(self, u):
        print("We don't want to call forward")
        # Step 3: Zero forcing term
        f = torch.zeros((self.size, self.size), dtype=self.dtype, device=self.device)

        # Step 4: Forward solve (batched or per sample)
        if u.ndim == 3:
            results = []
            for i in range(u.shape[0]):
                out = self.model(u[i], f).detach()
                results.append(out)
            return torch.stack(results)
        else:
            return self.model(u, f).detach()


    def plot_data(self, x, y, v, Jvp, file_path="plot.png", title="NS Sample Plot"):
        """
        Plot velocity fields and their curls for visualization.
        
        Args:
            x: Input velocity fields
            y: Output velocity fields
            v: Eigenvector velocity fields
            Jvp: Jacobian-vector product velocity fields
            file_path: Path to save the plot
        """
        def prepare_field(field):
            field = field.reshape(x.shape)
            return {
                'vorticity': field.cpu().numpy()
            }
        
        # Prepare all fields
        x_data = prepare_field(x)
        y_data = prepare_field(y)
        v_data = prepare_field(v)
        jvp_data = prepare_field(Jvp)
        
        # Create figure and subplots
        fig, axs = plt.subplots(4, 1, figsize=(5, 18))
        fig.suptitle(title)
        
        # Data to plot with corresponding titles
        plot_data = [
            (0, x_data, ['Input']),
            (1, y_data, ['Output']),
            (2, v_data, ['Eigenvector']),
            (3, jvp_data, ['Jvp'])
        ]
        
        # Plot all data
        for row, data, titles in plot_data:
            if row > 1:
                cmap = 'BuPu'
            else:
                cmap = 'jet' #'viridis'
            im = axs[row].imshow(data['vorticity'], cmap=cmap)
            if row == 1:
                # choose 10 levels between min and max of the field
                levels = np.linspace(data['vorticity'].min(), data['vorticity'].max(), 10)
                axs[row].contour(data['vorticity'], levels=levels, colors='coral', linewidths=2.0)
            fig.colorbar(im, ax=axs[row], fraction=0.046) 
            axs[row].set_title(titles[0])
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    @property
    def domain(self):
        return self.size * self.size

    @property
    def range(self):
        return self.size * self.size
