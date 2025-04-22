import torch
import numpy as np
import matplotlib.pyplot as plt
from groundwater.utils import GaussianRandomField, plot_fields
from groundwater.devito_op import GroundwaterModel
from .base import Simulator  # Assuming base class defines interface

class DarcySimulator(Simulator):
    def __init__(self, size=256, T=1.0, dtype=torch.float32):
        super().__init__()
        self.size = size
        self.dtype = dtype
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GroundwaterModel(size)
        self.T = T

    def sample(self):
        # Step 1: Sample from Gaussian Random Field
        grf = GaussianRandomField(2, self.size, alpha=2, tau=4)
        u_samples = grf.sample(1)
        
        return torch.tensor(u_samples[0], dtype=self.dtype, device=self.device)

    def forward(self, u):
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

    def plot_data(self, inputs, outputs, title="Darcy Simulator Results", file_path="darcy_plot.png"):
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().numpy()
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu().numpy()

        plot_fields(
            [np.exp(u) for u in inputs],
            [f"Input u(x) {i+1}" for i in range(len(inputs))],
            "Input Fields u(x)",
            contour=False,
        )

        plot_fields(
            outputs,
            [f"Output p(x) {i+1}" for i in range(len(outputs))],
            "Output Fields p(x)",
            contour=True,
        )
        plt.savefig(file_path)
        plt.close()

    @property
    def domain(self):
        return self.size * self.size

    @property
    def range(self):
        return self.size * self.size
