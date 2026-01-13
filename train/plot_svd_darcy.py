import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.linalg import LinearOperator, svds
from groundwater.devito_op import GroundwaterEquation
from groundwater.utils import GaussianRandomField

# ---------------- Configuration ----------------
DIM = 128
NUM_SAMPLES = 3
K_SINGULAR_VALUES = 1000
CSV_FILENAME = 'plot/singular_values_darcy.csv'
# -----------------------------------------------

def get_jacobian_svd(u_input, eq_model, f_np, k=100):
    """
    Computes top k singular values of the Jacobian at u_input.
    Uses Exact Linearization (Jv) and Adjoint (J.Tv).
    """
    # Ensure u is numpy for Devito
    if torch.is_tensor(u_input):
        u_np = u_input.detach().cpu().numpy().squeeze()
    else:
        u_np = u_input

    # 1. Precompute the Forward State (u -> p)
    # We need the full object (return_array=False) to pass to the adjoint/linearization
    p_fwd_obj = eq_model.eval_fwd_op(f_np, u_np, return_array=False)

    def mv(v):
        """
        Forward mode: J @ v
        Uses Devito's exact linearization instead of finite differences.
        """
        v_np = v.reshape(DIM, DIM).astype(np.float32)
        jvp = eq_model.compute_linearization(f_np, u_np, v_np)
        return jvp.reshape(-1)

    def rmv(v):
        """
        Adjoint mode: J.T @ v
        Uses Devito's adjoint operator.
        """
        v_np = v.reshape(DIM, DIM).astype(np.float32)
        # Here 'v' acts as the residual vector we are projecting back
        grad = eq_model.compute_gradient(u_np, v_np, p_fwd_obj)
        return grad.reshape(-1)

    # Define Linear Operator
    n_params = DIM * DIM
    J_op = LinearOperator((n_params, n_params), matvec=mv, rmatvec=rmv)

    print(f"   Computing top {k} SVs...")
    try:
        # svds returns singular values in ascending order
        _, s, _ = svds(J_op, k=k)
        return s[::-1] # Reverse to get descending order
    except Exception as e:
        print(f"   SVD convergence failed: {e}")
        return np.zeros(k)

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    # Initialize Physics Engine
    forcing_term = np.ones((DIM, DIM), dtype=np.float32)
    eq_model = GroundwaterEquation(DIM)

    # Initialize Random Field Generator
    grf = GaussianRandomField(2, DIM, alpha=2, tau=3.0)

    # Dictionary to store results for CSV
    results_data = {}

    plt.figure(figsize=(10, 6))

    for i in range(NUM_SAMPLES):
        print(f"--- Processing Sample {i+1}/{NUM_SAMPLES} ---")
        
        # 1. Sample a field
        u_sample = grf.sample(1)
        
        # 2. Binarize/Process
        u_sample[u_sample >= 0] = 0.9
        u_sample[u_sample < 0]  = 0.1
        u_np = u_sample[0].astype(np.float32)

        # 3. Compute Decay
        s_values = get_jacobian_svd(u_np, eq_model, forcing_term, k=K_SINGULAR_VALUES)
        
        # 4. Store in dictionary
        results_data[f'Sample_{i+1}'] = s_values

        # 5. Plot
        plt.plot(s_values, label=f'Sample {i+1}', linewidth=2, alpha=0.8)

    # ---------------- Save to CSV ----------------
    print(f"Saving singular values to {CSV_FILENAME}...")
    df = pd.DataFrame(results_data)
    df.to_csv(CSV_FILENAME, index_label="Rank")
    print("Save complete.")

    # ---------------- Visualization ----------------
    plt.yscale('log')
    plt.title(f'Singular Value Decay of Darcy Jacobian ({DIM}x{DIM})')
    plt.ylabel(r'Singular Value ($\sigma_i$)')
    plt.xlabel(r'Index $i$')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot/jacobian_svd_decay_darcy.png')
    print("Plot saved to 'plot/jacobian_svd_decay_darcy.png'.")
    plt.show()