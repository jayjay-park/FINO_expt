import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Adjust paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from groundwater.devito_op import GroundwaterEquation
    from datasets.simulators.NS import NavierStokesSimulator
    from groundwater.utils import GaussianRandomField
    from models.ns_inversion import NSModel 
except ImportError:
    print("Warning: Check your imports.")

# ---------------- Configuration ----------------
DIM_DARCY = 128
DIM_NS = 64
NUM_SAMPLES = 3
NUM_OBSERVATIONS = 400
PLOT_EXTEND = 50  # How many "zeros" to plot beyond m
NOISE_STD = 0.01  # For reference line
CSV_NAME = "plot/svd_dense_full.csv"

# Checkpoints
CKPT_DARCY = "checkpoints/Darcy_training.ckpt"
CKPT_NS    = "checkpoints/NS_training.ckpt"

# ---------------- Helper: Sparse Mask ----------------
def generate_sparse_mask(dim, num_obs):
    return np.random.choice(dim*dim, num_obs, replace=False)

# ---------------- 1. Dense Jacobian: Simulator ----------------
def get_dense_jacobian_sim(u_input, simulator, L, mode='NS'):
    """
    Constructs the full (m x N) Jacobian Matrix row-by-row.
    """
    if torch.is_tensor(u_input): u_np = u_input.detach().cpu().numpy().squeeze()
    else: u_np = u_input

    m = len(L)
    
    if mode == 'Darcy':
        dim = DIM_DARCY
        n = dim * dim
        f_np = np.ones((dim, dim), dtype=np.float32)
        # Precompute forward state once
        p_fwd_obj = simulator.eval_fwd_op(f_np, u_np, return_array=False)
        
        # J matrix
        J = np.zeros((m, n), dtype=np.float32)
        
        # Loop over observations (Rows of J)
        # Each row is the gradient of that single observation w.r.t input
        print(f"   Constructing Darcy Jacobian ({m} rows)...")
        for i in range(m):
            # Create a "sparse residual" with 1.0 at the sensor location
            v_full = np.zeros((dim, dim), dtype=np.float32)
            flat_idx = L[i]
            # Convert flat index to 2D
            r, c = divmod(flat_idx, dim)
            v_full[r, c] = 1.0
            
            # Compute Gradient (J.T * v) -> returns the row
            grad = simulator.compute_gradient(u_np, v_full, p_fwd_obj)
            J[i, :] = grad.reshape(-1)
            
    else: # NS
        dim = DIM_NS
        n = dim * dim
        # NS Simulator Wrapper supports Autograd
        u_ten = torch.tensor(u_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to("cuda")
        u_ten.requires_grad_(True)
        
        # Run forward once to build graph? No, need to retain graph or rerun.
        # Simplest stable way: Re-run forward or use functional.jacobian if memory permits.
        # Given m=200, a loop is memory-safe.
        
        print(f"   Constructing NS Jacobian ({m} rows)...")
        J = np.zeros((m, n), dtype=np.float32)
        
        # Pre-run to get output size
        out_baseline = simulator(u_ten) # Shape [1, 1, 64, 64]
        
        for i in range(m):
            if u_ten.grad is not None: u_ten.grad.zero_()
            
            # We want gradient of y[L[i]] w.r.t u
            # Run Forward
            out = simulator(u_ten)
            val = out.reshape(-1)[L[i]]
            
            # Backward
            val.backward()
            
            # Store Row
            J[i, :] = u_ten.grad.detach().cpu().numpy().reshape(-1)
            
    return J

# ---------------- 2. Dense Jacobian: Neural Network ----------------
def get_dense_jacobian_nn(u_input, model, L):
    """
    Constructs full (m x N) Jacobian for NN.
    """
    device = next(model.parameters()).device
    u_tensor = torch.tensor(u_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # We can use functional.jacobian for the subset of outputs
    # Wrapper function that returns only the observed pixels
    def forward_masked(x):
        out = model(x)
        return out.reshape(-1)[torch.tensor(L, device=device)]
    
    # Compute Jacobian (m x 1 x 1 x H x W)
    # This is usually fast enough for m=200
    print(f"   Computing NN Jacobian via Autograd...")
    J_tensor = torch.autograd.functional.jacobian(forward_masked, u_tensor)
    
    # Reshape to (m, N)
    return J_tensor.reshape(len(L), -1).detach().cpu().numpy()

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init Physics
    darcy_sim = GroundwaterEquation(DIM_DARCY)
    ns_sim = NavierStokesSimulator(DIM_NS, DIM_NS, 100, 2.0, 1e-3).to(device)
    
    # Init Models (Try/Except)
    try:
        model_darcy = NSModel.load_from_checkpoint(CKPT_DARCY, strict=False).eval().to(device)
        model_ns    = NSModel.load_from_checkpoint(CKPT_NS, strict=False).eval().to(device)
    except:
        model_darcy, model_ns = None, None
        print("Skipping NNs (Checkpoints not found)")

    # Data Containers
    results = {}
    
    grf_darcy = GaussianRandomField(2, DIM_DARCY, alpha=2.0, tau=3.0)
    grf_ns    = GaussianRandomField(2, DIM_NS, alpha=2.5, tau=3.0)

    # --- Processing ---
    for i in range(NUM_SAMPLES):
        print(f"\nSample {i+1}/{NUM_SAMPLES}")
        
        # 1. Darcy
        u = grf_darcy.sample(1)[0].astype(np.float32)
        L = generate_sparse_mask(DIM_DARCY, NUM_OBSERVATIONS)
        
        # Sim
        J_sim = get_dense_jacobian_sim(u, darcy_sim, L, mode='Darcy')
        s_sim = np.linalg.svd(J_sim, compute_uv=False) # Returns m values
        results[f'Darcy_Sim_{i}'] = s_sim
        
        # NN
        if model_darcy:
            J_nn = get_dense_jacobian_nn(u, model_darcy, L)
            s_nn = np.linalg.svd(J_nn, compute_uv=False)
            results[f'Darcy_NN_{i}'] = s_nn

        # 2. NS
        u_ns = grf_ns.sample(1)[0].astype(np.float32)
        L_ns = generate_sparse_mask(DIM_NS, NUM_OBSERVATIONS)
        
        # Sim
        J_ns = get_dense_jacobian_sim(u_ns, ns_sim, L_ns, mode='NS')
        s_ns = np.linalg.svd(J_ns, compute_uv=False)
        results[f'NS_Sim_{i}'] = s_ns
        
        # NN
        if model_ns:
            J_nn_ns = get_dense_jacobian_nn(u_ns, model_ns, L_ns)
            s_nn_ns = np.linalg.svd(J_nn_ns, compute_uv=False)
            results[f'NS_NN_{i}'] = s_nn_ns

    # Save
    pd.DataFrame(results).to_csv(CSV_NAME)
    print("Computation Complete.")

    # --- Plotting with Padding ---
    plt.figure(figsize=(12, 6))
    
    # We want to plot x-axis from 0 to m + PLOT_EXTEND
    x_full = np.arange(NUM_OBSERVATIONS + PLOT_EXTEND)
    
    def plot_padded(keyword, label, color, style):
        cols = [c for c in results.keys() if keyword in c]
        if not cols: return
        
        # Get matrix of shape (m, num_samples)
        data_matrix = np.array([results[c] for c in cols]).T 
        
        # Pad with zeros (or small epsilon for log plot)
        # Using 1e-6 as "numerical zero" for log plot visibility
        pad = np.full((PLOT_EXTEND, data_matrix.shape[1]), 1e-16)
        data_padded = np.vstack([data_matrix, pad])
        
        mean = data_padded.mean(axis=1)
        std = data_padded.std(axis=1)
        
        plt.plot(x_full, mean, label=label, color=color, linestyle=style, linewidth=2)
        plt.fill_between(x_full, mean-std, mean+std, color=color, alpha=0.1)

    plot_padded("Darcy_Sim", "Darcy Simulator", "tab:blue", "-")
    plot_padded("NS_Sim",    "NS Simulator",    "tab:orange", "-")
    plot_padded("Darcy_NN",  "Darcy NN",        "tab:blue", "--")
    plot_padded("NS_NN",     "NS NN",           "tab:orange", "--")

    # Annotations
    plt.axhline(y=NOISE_STD, color='red', linestyle='--', label="Noise Floor")
    plt.axvline(x=NUM_OBSERVATIONS, color='gray', linestyle=':', label="Num Observations (m)")
    
    plt.yscale('log')
    plt.title(f"SVD Decay with Sparse Observations (o={NUM_OBSERVATIONS})")
    plt.xlabel("Singular Value Index")
    plt.ylabel(r"Singular Value $\sigma_i$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adjust ylim to show the drop off
    plt.ylim(bottom=1e-5) 
    
    plt.savefig("plot/svd_dense_full_decay.png")
    plt.show()