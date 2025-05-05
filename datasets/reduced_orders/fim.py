import torch
from .base import ReducedModel
import matplotlib.pyplot as plt
import threading
import numpy as np


class FIMReducedModel(ReducedModel):
    def __init__(self, eigen_value_fraction, eigen_vector_count, simulator_type, noise_std, probe_size):
        self.eigen_value_fraction = eigen_value_fraction
        self.eigen_vector_count = eigen_vector_count
        self.probe_size = probe_size   # number of columns m per probe
        self.thread_lock = threading.Lock()
        self.simulator_type = simulator_type
        self.noise_std = noise_std

    # def get_direction(self, simulator, x):
    #     """
    #     Compute averaged score matrix Q over a batch of inputs x
    #     via column‐wise jittered subsampling.
    #     """
    #     with self.thread_lock:
    #         # x: [num_sub, p] or [num_sub, d, d]
    #         num_sub = x.shape[0]
    #         p = simulator.domain
    #         d = int(np.sqrt(p))
    #         r = self.eigen_vector_count + 5

    #         # 1) jittered sample fixed columns L from [0..d-1]
    #         m = self.probe_size
    #         bins = torch.linspace(0, d, m+1, device=x.device)
    #         L = []
    #         for i in range(m):
    #             start, end = int(bins[i].item()), int(bins[i+1].item())
    #             L.append(torch.randint(start, max(start+1,end), (), device=x.device).item())

    #         # accumulator for Q
    #         Q_acc = torch.zeros(p, r, device=x.device)

    #         # loop over each sample in the batch
    #         for b in range(num_sub):
    #             xb = x[b]
    #             # forward + pullback setup for this sample
    #             if self.simulator_type != "DARCY":
    #                 yb, vjp_func = torch.func.vjp(simulator, xb)
    #             else:
    #                 xb_np = xb.cpu().numpy()
    #                 f = np.zeros_like(xb_np)
    #                 p_fwd = simulator.model.eval_fwd_op(f, xb_np, return_array=False)

    #             Qb = torch.zeros(p, r, device=x.device)
    #             # do r probes on this one input
    #             for j in range(r):
    #                 print("rank ", j)
    #                 # noise per column
    #                 eps = torch.randn(d, m, device=x.device) * self.noise_std
    #                 weights = eps / (self.noise_std**2)  # shape [d,m]

    #                 # build 2D probe and flatten
    #                 v2d = torch.zeros((d, d), device=x.device)
    #                 v2d[:, L] = weights
    #                 vj = v2d.reshape(-1)

    #                 # compute score
    #                 if self.simulator_type != "DARCY":
    #                     qj = vjp_func(vj)[0]
    #                 else:
    #                     probe_vec = v2d.cpu().numpy()
    #                     grad = simulator.model.compute_gradient(xb_np, probe_vec, p_fwd)
    #                     qj = torch.tensor(grad, device=x.device).reshape(-1)

    #                 Qb[:, j] = qj

    #             Q_acc += Qb

    #         # average over all subsamples
    #         Q_avg = Q_acc / float(num_sub)

    #         return Q_avg, L

    def get_direction(self, simulator, x):
        """
        Compute top-FIM modes via column-wise subsampling with a fixed set of columns,
        and shade those columns on the plotted mode.
        """
        with self.thread_lock:
            p = simulator.domain
            d = int(np.sqrt(p))
            r = self.eigen_vector_count
            m = self.probe_size

            # 1) sample fixed column‐subset L from [0..d-1]
            # L = torch.randperm(d, device=x.device)[:m].tolist()  # columns indices
            # no more flatten-based coords needed
            # 1) jittered sampling of columns
            bins = torch.linspace(0, d, m+1, device=x.device)
            L = []
            for i in range(m):
                s, e = int(bins[i].item()), int(bins[i+1].item())
                L.append(torch.randint(s, max(s+1,e), (1,), device=x.device).item())
            # proceed with v2d[:, L] = weights …


            # forward + pullback setup
            if self.simulator_type != "DARCY":
                y, vjp_func = torch.func.vjp(simulator, x)
            else:
                x_np = x.cpu().numpy()
                f = np.zeros_like(x_np)
                p_fwd = simulator.model.eval_fwd_op(f, x_np, return_array=False)

            Q = torch.zeros(p, r, device=x.device)

            for j in range(r):
                # 2) noise per column
                eps = torch.randn(d, m, device=x.device) * self.noise_std  # one per cell in each column
                weights = eps / (self.noise_std**2)                          # shape [d,m]

                # 3) build 2D probe v2d and flatten
                v2d = torch.zeros((d, d), device=x.device)
                v2d[:, L] = weights  # assign entire columns
                v_j = v2d.reshape(-1)  

                # 4) compute score q_j
                if self.simulator_type != "DARCY":
                    q_j = vjp_func(v_j)[0]
                else:
                    probe_vec = v2d.cpu().numpy()
                    grad = simulator.model.compute_gradient(x_np, probe_vec, p_fwd)
                    q_j = torch.tensor(grad, device=x.device).reshape(-1)

                Q[:, j] = q_j

            # 5) SVD
            U, S, Vh = torch.linalg.svd(Q, full_matrices=False)

            # 6) plot leading mode and shade selected columns
            mode1 = U[:, 0].reshape(d, d).cpu().numpy()
            plt.figure(figsize=(5, 5))
            plt.imshow(mode1, cmap='BuPu', origin='lower')
            plt.title("Leading FIM Mode (fixed columns)")
            plt.colorbar(fraction=0.046)

            # shade each selected column
            for col in L:
                plt.axvspan(col - 0.5, col + 0.5, color='gray', alpha=0.3)

            plt.savefig('FIM_mode1_shaded_columns.png', bbox_inches='tight')
            plt.close()

            eigvecs = U[:, :r]
            eigvals  = S[:r]**2
            return eigvecs, eigvals


    
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

