import torch
from .base import ReducedModel
import matplotlib.pyplot as plt
import threading
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor


class FIMReducedModel(ReducedModel):
    def __init__(self, eigen_value_fraction, eigen_vector_count, simulator_type, noise_std, probe_size):
        self.eigen_value_fraction = eigen_value_fraction
        self.eigen_vector_count = eigen_vector_count
        self.probe_size = probe_size   # number of columns m per probe
        self.thread_lock = threading.Lock()
        self.simulator_type = simulator_type
        self.noise_std = noise_std

    def get_direction(self, simulator, x):
        with self.thread_lock:
            eigen_count = self.eigen_count(simulator)
            eigenvectors = torch.randn((simulator.domain, eigen_count))
            if self.simulator_type != "DARCY":
                y, vjp_func = torch.func.vjp(simulator, x)
            
            Z = torch.randn((simulator.range, eigen_count)).to(x.device)
            B, R = torch.linalg.qr(Z)

            if self.simulator_type == "DARCY":
                x = x.cpu().numpy()
                # Forcing term (e.g., external influences) is initialized as zeros
                f = np.zeros(x.shape)
                p_fwd = simulator.model.eval_fwd_op(f, x, return_array=False)
            
            Q = torch.zeros((simulator.domain, eigen_count))
            for j in range(eigen_count):
                print(f"Computing FIM Eigenvector {j + 1} of {eigen_count}")

                if self.simulator_type == "DARCY":
                    probe_vector = B[:, j].reshape(x.shape).cpu().numpy()
                    gradient = simulator.model.compute_gradient(x, probe_vector, p_fwd)
                    Q[:, j] = torch.tensor(gradient).reshape((simulator.domain,))

                    plt.figure(figsize=(5, 5))
                    plt.imshow(gradient, cmap='viridis')
                    plt.title(r'$J^Tv$')
                    plt.colorbar()
                    plt.savefig(f'Devito_vjp={j}.png', bbox_inches='tight')
                    plt.close()

                else:
                    probe_vector = B[:, j].reshape(y.shape)
                    Q[:, j] = vjp_func(probe_vector)[0].reshape((simulator.domain,))

            U, S, V = torch.linalg.svd(Q)
            plt.figure(figsize=(5, 5))
            plt.imshow(U[:, 0].reshape(128,128), cmap='viridis')
            plt.title(r'$U,S,V^T = J^T\Sigma^{-1}J$')
            plt.colorbar()
            plt.savefig(f'Devito_eig_1.png', bbox_inches='tight')
            plt.close()
            eigenvectors = U[:, :eigen_count]
            
            return eigenvectors, S

    def _compute_single_grad(self, simulator, x_np, probe_2d, p_fwd, simulator_type, i, vjp_func):
        """
        Run in a worker thread. Returns a flat gradient of shape (p,).
        """
        if simulator_type == "DARCY":
            g = simulator.model.compute_gradient(x_np, probe_2d, p_fwd).reshape(-1)  # [d,d]
        else:
            print("i", i)
            probe_2d_gpu = torch.tensor(probe_2d)
            g = vjp_func(probe_2d_gpu)[0].reshape(-1)
            print("g", g.shape)
        return g


    def compute_score_matrix(self, simulator, x, L):
        """
        Thread-parallel Darcy fallback. Returns Qb of shape [p, r] on CUDA.
        """
        # 1) problem sizes
        p = simulator.domain              # = d*d
        d = int(np.sqrt(p))
        r = self.eigen_vector_count + 5
        m = self.probe_size               # == len(L)

        # 2) build your GPU probe matrix [r, p]
        idx = torch.tensor(L, dtype=torch.long, device='cuda')  
        eps = torch.randn(r, m, device='cuda') * self.noise_std  
        weights = eps / (self.noise_std**2)                     

        rows = torch.arange(r, device='cuda').unsqueeze(1)      
        probe_flat = torch.zeros((r, p), device='cuda')        
        probe_flat[rows, idx] = weights                        

        # 3) one Darcy forward solve
        x_np, p_fwd = None, None
        if self.simulator_type == "DARCY":
            x_np = x
            f    = np.zeros_like(x_np)
            p_fwd = simulator.model.eval_fwd_op(
                f, x_np, simulator.T, return_array=False
            )
        else:
            x_np = torch.tensor(x).cuda()
            print("before")
            _, vjp_func = torch.func.vjp(simulator, x_np)
            print("after")

        # 4) prepare numpy probes for each thread
        probes_np = probe_flat.cpu().numpy().reshape(r, d, d)  # [r, d, d]

        # 5) thread-parallel compute_gradient calls
        grads_np = np.empty((r, p), dtype=np.float32)
        with ThreadPoolExecutor() as exe:
            futures = {
                exe.submit(self._compute_single_grad, simulator, x_np, probes_np[i], p_fwd, self.simulator_type, i, vjp_func): i
                for i in range(r)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                grads_np[i] = fut.result()

        # 6) back to GPU, shape [p, r]
        Qb_rows = torch.from_numpy(grads_np).to('cuda')  # [r, p]
        Qb      = Qb_rows.transpose(0, 1)               # [p, r]
        return Qb


    
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

