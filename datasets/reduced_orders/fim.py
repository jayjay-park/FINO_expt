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


    # def compute_score_matrix(self, simulator, x, L):
    #     """
    #     Matrix‑free point‑wise jittered subsampled FIM → Qb [p×r].
    #     simulator:     your forward function
    #     x:             current input state, shape [d, d]
    #     L:             list of m flat indices in [0, p)
    #     """
    #     p = simulator.domain              # = d*d
    #     d = int(np.sqrt(p))
    #     r = self.eigen_vector_count + 5
    #     m = self.probe_size               # must equal len(L)

    #     # prepare pullback
    #     if self.simulator_type != "DARCY":
    #         _, vjp_func = torch.func.vjp(simulator, x)
    #     else:
    #         x_np = x
    #         f = np.zeros_like(x_np)
    #         # precompute forward evaluation once
    #         p_fwd = simulator.model.eval_fwd_op(f, x_np, simulator.T, return_array=False)

    #     Qb = torch.zeros(p, r, device='cuda')

    #     for j in range(r):
    #         print("r", j)
    #         # 1) one noise per sampled point
    #         eps     = torch.randn(m, device='cuda') * self.noise_std   # [m]
    #         weights = eps / (self.noise_std**2)                          # [m]

    #         # 2) scatter into a d×d probe array
    #         v2d = torch.zeros((d, d), device='cuda')
    #         for k, idx in enumerate(L):
    #             i = idx // d
    #             jcol = idx % d
    #             v2d[i, jcol] = weights[k]

    #         v_j = v2d.reshape(-1)   # [p]

    #         # 3) pull back to get the jth score vector
    #         if self.simulator_type != "DARCY":
    #             q_j = vjp_func(v_j)[0]                        # [p]
    #         else:
    #             probe_np = v2d.cpu().numpy()
    #             grad = simulator.model.compute_gradient(
    #                     x_np, probe_np, p_fwd
    #                 )                                     # [d,d]
    #             q_j = torch.tensor(grad, device='cuda').reshape(-1)

    #         Qb[:, j] = q_j

    #     return Qb


    # from torch.func import vjp, vmap

    # def compute_score_matrix(self, simulator, x, L):
    #     """
    #     Vectorized, batched version of your Qb computation.
    #     Returns Qb of shape [p, r].
    #     """
    #     # 1) problem size & rank
    #     p = simulator.domain           # = d*d
    #     d = int(np.sqrt(p))
    #     r = self.eigen_vector_count + 5
    #     m = self.probe_size            # == len(L)

    #     # 2) precompute L as a tensor for fast indexing
    #     #    shape [m], dtype long, on CUDA
    #     idx = torch.tensor(L, dtype=torch.long, device='cuda')  

    #     # 3) prepare the VJP function once
    #     if self.simulator_type != "DARCY":
    #         # simulator: x -> y
    #         _, vjp_func = vjp(simulator, x)
    #     else:
    #         # keep your Darcy CPU‐path as is
    #         x_np = x
    #         f   = np.zeros_like(x_np)
    #         p_fwd = simulator.model.eval_fwd_op(f, x_np, simulator.T, return_array=False)

    #     # 4) sample all epsilons at once: shape [r, m]
    #     eps = torch.randn(r, m, device='cuda') * self.noise_std
    #     weights = eps / (self.noise_std**2)          # [r, m]

    #     # 5) build a batched flat probe‐matrix of shape [r, p]
    #     #    rows = 0..r-1, cols = the m positions in L
    #     rows = torch.arange(r, device='cuda').unsqueeze(1)  # [r,1]
    #     probe_flat = torch.zeros((r, p), device='cuda')
    #     probe_flat[rows, idx] = weights                   # broadcast [r,m]


    #     # 6) run a single batched VJP to get [r, p] of score-vectors
    #     if self.simulator_type != "DARCY":
    #         # vjp_func returns a tuple—even in batched mode—so we pick [0]
    #         Qb_rows = vmap(lambda v: vjp_func(v)[0])(probe_flat)  # [r, p]
    #         print("Qb_rows.shape =", Qb_rows.shape)
    #     else:
    #         # fallback to your CPU‐based Darcy gradient
    #         # (we could also vmap the compute_gradient call here)
    #         grads = []
    #         for j in range(r):
    #             probe_2d = probe_flat[j].reshape(d, d).cpu().numpy()
    #             g = simulator.model.compute_gradient(x_np, probe_2d, p_fwd)  # [d,d]
    #             grads.append(torch.tensor(g, device='cuda').reshape(-1))
    #         Qb_rows = torch.stack(grads, dim=0)  # [r, p]

    #     # 7) transpose so we match your original [p, r]
    #     Qb = Qb_rows.transpose(0, 1)  # [p, r]

    #     return Qb


    def _compute_single_grad(self, simulator, x_np, probe_2d, p_fwd):
        """
        Run in a worker thread. Returns a flat gradient of shape (p,).
        """
        g = simulator.model.compute_gradient(x_np, probe_2d, p_fwd)  # [d,d]
        return g.reshape(-1)


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

        # 4) prepare numpy probes for each thread
        probes_np = probe_flat.cpu().numpy().reshape(r, d, d)  # [r, d, d]

        # 5) thread-parallel compute_gradient calls
        grads_np = np.empty((r, p), dtype=np.float32)
        with ThreadPoolExecutor() as exe:
            futures = {
                exe.submit(self._compute_single_grad, simulator, x_np, probes_np[i], p_fwd): i
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

