import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Adjust paths as needed
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
NUM_OBSERVATIONS = 3000  # m
CSV_NAME = "plot/svd_exact_sparse.csv"

# Checkpoint paths (Update if necessary)
CKPT_DARCY = "checkpoints/Darcy_training.ckpt"
CKPT_NS    = "checkpoints/NS_training.ckpt"

# ---------------- Helper: Sparse Mask ----------------
def generate_sparse_mask(dim, num_obs):
    """Randomly selects 'num_obs' indices from the flattened grid."""
    return np.random.choice(dim*dim, num_obs, replace=False)

# ---------------- 1. Exact Jacobian: Simulator ----------------
def get_exact_jacobian_sim(u_input, simulator, L, mode='NS'):
    """
    Computes the exact m x N Jacobian of the observation operator.
    Each row i is the gradient of the i-th observation w.r.t the input u.
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
        
        J = np.zeros((m, n), dtype=np.float32)
        print(f"   Computing Darcy Jacobian ({m} rows)...")
        
        for i in range(m):
            # To get gradient of y_i, we pass a "delta" residual at that location
            v_full = np.zeros((dim, dim), dtype=np.float32)
            idx = L[i]
            r, c = divmod(idx, dim)
            v_full[r, c] = 1.0
            
            # Compute Gradient: grad_u ( y_i )
            grad = simulator.compute_gradient(u_np, v_full, p_fwd_obj)
            J[i, :] = grad.reshape(-1)
            
    else: # NS
        dim = DIM_NS
        n = dim * dim
        # Wrap input for Autograd
        u_ten = torch.tensor(u_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to("cuda")
        u_ten.requires_grad_(True)
        
        J = np.zeros((m, n), dtype=np.float32)
        print(f"   Computing NS Jacobian ({m} rows)...")
        
        for i in range(m):
            if u_ten.grad is not None: u_ten.grad.zero_()
            
            # Forward pass
            out = simulator(u_ten) # Shape [1, 1, 64, 64]
            
            # Select the single observation pixel
            val = out.reshape(-1)[L[i]]
            
            # Backward pass (computes row of Jacobian)
            val.backward()
            
            J[i, :] = u_ten.grad.detach().cpu().numpy().reshape(-1)
            
    return J

# ---------------- 2. Exact Jacobian: Neural Network ----------------
def get_exact_jacobian_nn(u_input, model, L):
    """
    Computes exact m x N Jacobian for NN using Autograd.
    """
    device = next(model.parameters()).device
    u_tensor = torch.tensor(u_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Define a function that returns ONLY the observed values
    def forward_masked(x):
        out = model(x)
        return out.reshape(-1)[torch.tensor(L, device=device)]
    
    print(f"   Computing NN Jacobian via Autograd...")
    # torch.autograd.functional.jacobian computes the exact matrix
    # Output shape: (m, 1, 1, H, W) -> Flatten to (m, N)
    J_tensor = torch.autograd.functional.jacobian(forward_masked, u_tensor)
    
    return J_tensor.reshape(len(L), -1).detach().cpu().numpy()

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Init Physics
    darcy_sim = GroundwaterEquation(DIM_DARCY)
    ns_sim = NavierStokesSimulator(DIM_NS, DIM_NS, 100, 2.0, 1e-3).to(device)
    
    # Init Models
    try:
        model_darcy = NSModel.load_from_checkpoint(CKPT_DARCY, strict=False).eval().to(device)
        model_ns    = NSModel.load_from_checkpoint(CKPT_NS, strict=False).eval().to(device)
    except:
        model_darcy, model_ns = None, None
        print("Note: NN checkpoints not found. Skipping NN plots.")

    results = {}
    
    # Gaussian Random Fields (Continuous)
    grf_darcy = GaussianRandomField(2, DIM_DARCY, alpha=2.0, tau=3.0)
    grf_ns    = GaussianRandomField(2, DIM_NS, alpha=2.5, tau=3.0)

    # --- Run Loop ---
    for i in range(NUM_SAMPLES):
        print(f"\nProcessing Sample {i+1}/{NUM_SAMPLES}...")
        
        # 1. Darcy
        u_darcy = grf_darcy.sample(1)[0].astype(np.float32) # Standard GRF (No binarization)
        L_darcy = generate_sparse_mask(DIM_DARCY, NUM_OBSERVATIONS)
        
        # Sim
        J_sim = get_exact_jacobian_sim(u_darcy, darcy_sim, L_darcy, mode='Darcy')
        results[f'Darcy_Sim_{i}'] = np.linalg.svd(J_sim, compute_uv=False)
        
        # NN
        if model_darcy:
            J_nn = get_exact_jacobian_nn(u_darcy, model_darcy, L_darcy)
            results[f'Darcy_NN_{i}'] = np.linalg.svd(J_nn, compute_uv=False)

        # 2. NS
        u_ns = grf_ns.sample(1)[0].astype(np.float32)
        L_ns = generate_sparse_mask(DIM_NS, NUM_OBSERVATIONS)
        
        # Sim
        J_ns_sim = get_exact_jacobian_sim(u_ns, ns_sim, L_ns, mode='NS')
        results[f'NS_Sim_{i}'] = np.linalg.svd(J_ns_sim, compute_uv=False)
        
        # NN
        if model_ns:
            J_nn_ns = get_exact_jacobian_nn(u_ns, model_ns, L_ns)
            results[f'NS_NN_{i}'] = np.linalg.svd(J_nn_ns, compute_uv=False)

    # Save Data
    pd.DataFrame(results).to_csv(CSV_NAME)
    print("Computation Complete.")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    colors = {"Darcy": "tab:blue", "NS": "tab:orange"}
    styles = {"Sim": "-", "NN": "--"}
    
    # Plot curves
    for key in results:
        data = results[key]
        kind = "Darcy" if "Darcy" in key else "NS"
        source = "Sim" if "Sim" in key else "NN"
        
        # Determine Label for legend (only label first instance)
        lbl = f"{kind} {source}"
        if lbl in plt.gca().get_legend_handles_labels()[1]:
            lbl = None
            
        plt.plot(data, color=colors[kind], linestyle=styles[source], 
                 alpha=0.7, linewidth=1.5, label=lbl)

    plt.yscale('log')
    plt.title(f"SVD of Observation Jacobian (m={NUM_OBSERVATIONS})\nGradient $\\nabla_a(y)$")
    plt.xlabel("Index")
    plt.ylabel(r"Singular Value $\sigma_i$")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plot/svd_exact_sparse_decay.png")
    plt.show()