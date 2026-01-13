import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.linalg import LinearOperator, svds
import sys
import os

# Adjust path to find the 'datasets' module if necessary
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.simulators.NS import NavierStokesSimulator
from groundwater.utils import GaussianRandomField

# ---------------- Configuration ----------------
DIM = 64
NUM_SAMPLES = 3
K_SINGULAR_VALUES = 1000
CSV_FILENAME = 'plot/singular_values_ns.csv'
FD_EPSILON = 1e-4   # Perturbation size for Jv
# -----------------------------------------------

def get_ns_jacobian_svd(u_input, model, k=100):
    """
    Computes top k singular values of the NS Jacobian at u_input.
    Uses:
      1. Finite Difference for Forward Mode (J @ v)
      2. PyTorch Autograd for Adjoint Mode (J.T @ v)
    """
    device = model.device
    
    # Prepare input tensor (1, 1, 64, 64)
    if not torch.is_tensor(u_input):
        u_tensor = torch.tensor(u_input, dtype=torch.float32, device=device)
    else:
        u_tensor = u_input.to(device)
        
    if u_tensor.ndim == 2:
        u_tensor = u_tensor.unsqueeze(0).unsqueeze(0)
    elif u_tensor.ndim == 3:
        u_tensor = u_tensor.unsqueeze(0)
    
    # We need gradients w.r.t input for J^T v
    u_tensor.requires_grad_(True)

    # Precompute baseline forward pass
    # Output shape likely (1, 1, 64, 64) based on your inversion code
    p_baseline = model(u_tensor)
    
    def mv(v):
        """
        Forward mode: J @ v
        Implemented via Finite Difference: (F(u + eps*v) - F(u)) / eps
        """
        # Prepare perturbation vector v
        v_in = torch.tensor(v.reshape(u_tensor.shape), dtype=torch.float32, device=device)
        
        with torch.no_grad():
            u_pert = u_tensor + FD_EPSILON * v_in
            p_pert = model(u_pert)
            jvp = (p_pert - p_baseline) / FD_EPSILON
            
        return jvp.detach().cpu().numpy().reshape(-1)

    def rmv(v):
        """
        Adjoint mode: J.T @ v
        Implemented via PyTorch Autograd (Standard VJP)
        """
        # Clean gradients
        if u_tensor.grad is not None:
            u_tensor.grad.zero_()
            
        # Prepare vector v (this is the gradient output we project back)
        v_out = torch.tensor(v.reshape(p_baseline.shape), dtype=torch.float32, device=device)
        
        # Compute VJP: grad(outputs=p_baseline, inputs=u_tensor, grad_outputs=v_out)
        # retain_graph=True is crucial because we reuse p_baseline for every iteration
        grad = torch.autograd.grad(
            outputs=p_baseline, 
            inputs=u_tensor, 
            grad_outputs=v_out, 
            retain_graph=True
        )[0]
        
        return grad.detach().cpu().numpy().reshape(-1)

    # Define Linear Operator
    n_params = DIM * DIM # 64*64 = 4096
    J_op = LinearOperator((n_params, n_params), matvec=mv, rmatvec=rmv)

    print(f"   Computing top {k} SVs using PyTorch Autograd...")
    try:
        # svds returns singular values in ascending order
        _, s, _ = svds(J_op, k=k)
        return s[::-1] # Reverse to get descending order
    except Exception as e:
        print(f"   SVD convergence failed: {e}")
        return np.zeros(k)

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize NS Simulator (Wrapper)
    # Based on inversion (12).py: NavierStokesSimulator(dim, dim, 100, 2.0, 1e-3)
    print(f"Initializing NavierStokesSimulator ({DIM}x{DIM})...")
    ns_model = NavierStokesSimulator(DIM, DIM, 100, 2.0, 1e-3).to(device)
    
    # Ensure model is in eval mode if it has layers (though usually simulator is functional)
    # But we need autograd, so we keep standard behavior.
    
    # Initialize Random Field Generator
    # We generate 64x64 fields
    grf = GaussianRandomField(2, DIM, alpha=2.5, tau=3.0)

    results_data = {}
    plt.figure(figsize=(10, 6))

    for i in range(NUM_SAMPLES):
        print(f"--- Processing Sample {i+1}/{NUM_SAMPLES} ---")
        
        # 1. Sample input
        u_sample = grf.sample(1)
        u_np = u_sample[0].astype(np.float32)

        # 2. Compute Decay
        s_values = get_ns_jacobian_svd(u_np, ns_model, k=K_SINGULAR_VALUES)
        
        # 3. Store and Plot
        results_data[f'Sample_{i+1}'] = s_values
        plt.plot(s_values, label=f'Sample {i+1}', linewidth=2, alpha=0.8)

    # ---------------- Save to CSV ----------------
    print(f"Saving singular values to {CSV_FILENAME}...")
    df = pd.DataFrame(results_data)
    df.to_csv(CSV_FILENAME, index_label="Rank")
    print("Save complete.")

    # ---------------- Visualization ----------------
    plt.yscale('log')
    plt.title(f'Singular Value Decay of NS Jacobian ({DIM}x{DIM})')
    plt.ylabel(r'Singular Value ($\sigma_i$)')
    plt.xlabel(r'Index $i$')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot/jacobian_svd_decay_ns.png')
    print("Plot saved to 'plot/ns_jacobian_svd_decay.png'.")
    # plt.show() # Comment out if running on headless server