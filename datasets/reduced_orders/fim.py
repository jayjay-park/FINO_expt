import torch
from .base import ReducedModel
import matplotlib.pyplot as plt
import threading
import numpy as np

class FIMReducedModel(ReducedModel):
    def __init__(self, eigen_value_fraction, eigen_vector_count, simulator_type, noise_std):
        self.eigen_value_fraction = eigen_value_fraction
        self.eigen_vector_count = eigen_vector_count
        self.thread_lock = threading.Lock()
        self.simulator_type = simulator_type
        self.noise_std = noise_std

    def get_direction(self, simulator, x):
        """
        Compute top-FIM modes via random subsampling.
        Plots the first mode and marks all probe locations.
        """
        with self.thread_lock:
            # assume simulator.domain == p == d*d, simulator.range == p
            p = simulator.domain
            d = int(np.sqrt(p))
            r = self.eigen_vector_count + 5

            # forward pass + pullback setup
            if self.simulator_type != "DARCY":
                y, vjp_func = torch.func.vjp(simulator, x)  # y: [p], x: [p]
            else:
                x_np = x.cpu().numpy()
                f = np.zeros_like(x_np)
                p_fwd = simulator.model.eval_fwd_op(f, x_np, return_array=False)

            Q = torch.zeros(p, r, device=x.device)
            probe_coords = []

            for j in range(r):
                print(f"Computing probe {j+1}/{r}…")

                # 1) choose a random coordinate
                k = torch.randint(0, p, (1,)).item()       # integer in [0, p)
                i_row = k // d                            # row index
                j_col = k % d                             # col index
                probe_coords.append((i_row, j_col))       # for plotting (x=j_col, y=i_row)

                # 2) sample scalar noise and weight
                eps = torch.randn((), device=x.device) * self.noise_std
                w = eps / (self.noise_std**2)        # = σ^{-2} * ε

                # 3) form probe vector and do pull‑back
                if self.simulator_type != "DARCY":
                    e_k = torch.zeros(p, device=x.device)
                    e_k[k] = w
                    q_j = vjp_func(e_k)[0]                # shape [p]
                else:
                    # probe_vec = w.cpu().numpy().reshape(d, d)  # shape [d,d]
                    probe_vec = np.zeros((d, d), dtype=float)
                    probe_vec[i_row, j_col] = w.cpu().item()

                    grad = simulator.model.compute_gradient(
                        x_np, probe_vec, p_fwd
                    )  # returns array [d,d]
                    q_j = torch.tensor(grad, device=x.device).reshape(-1)

                Q[:, j] = q_j

            # 4) SVD
            U, S, Vh = torch.linalg.svd(Q, full_matrices=False)

            # 5) plot the leading mode + probe locations
            mode1 = U[:, 0].reshape(d, d).cpu().numpy()
            plt.figure(figsize=(5,5))
            plt.imshow(mode1, cmap='BuPu')
            plt.title("The Leading FIM Eigenvector")
            plt.colorbar(fraction=0.046)
            # overlay all probe points
            xs, ys = zip(*probe_coords)
            plt.scatter(xs, ys, s=30, facecolors='none', edgecolors='black')
            plt.legend(loc='upper right')
            plt.savefig('FIM_mode1_with_probes.png', bbox_inches='tight')
            plt.close()

            # return the top-r eigenvectors and singular values
            eigvecs = U[:, :self.eigen_vector_count]    # shape [p, r]
            return eigvecs, S[:self.eigen_vector_count]
    
    def plot_decay(self, s, path, title):
        plt.plot(s)
        plt.title(title)
        plt.xlabel("Eigenvector")
        plt.ylabel("Singular Value")
        plt.savefig(path)
        plt.close()

    def eigen_count(self, simulator):
        if self.eigen_vector_count is not None:
            return self.eigen_vector_count
        return int(simulator.domain * self.eigen_value_fraction)

