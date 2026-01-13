import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.linalg import LinearOperator, svds
import sys
import os
import yaml

# Adjust paths to match your directory structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ns_inversion import NSModel
from groundwater.utils import GaussianRandomField

# ---------------- Configuration ----------------
K_SINGULAR_VALUES = 1000
NUM_SAMPLES = 3

# Checkpoint Paths (UPDATE THESE PATHS based on your local files)
CKPT_DARCY = "checkpoints/checkpoints/n=400_e=64_m=FNO_s=RFS_l=JAC_20260102_152517/n=400_e=64_m=FNO_s=RFS_l=JAC_epoch=199_val_rel_l2_loss=0.0095.ckpt"
CKPT_NS    = "checkpoints/n=400_e=400_m=FNO_s=RFS_l=JAC_20251015_130613/n=400_e=400_m=FNO_s=RFS_l=JAC_epoch=210_val_rel_l2_loss=0.0591.ckpt"
CKPT_DARCY_MSE = "checkpoints/Darcy_training_20251101_214307/Darcy_training_epoch=172_val_rel_l2_loss=0.0096.ckpt"
CKPT_NS_MSE = "checkpoints/n=400_m=FNO_l=L2_20251015_125349/n=400_m=FNO_l=L2_epoch=381_val_rel_l2_loss=0.0594.ckpt"
TYPE = "JAC"

# Dimensions
DIM_DARCY = 128
DIM_NS    = 64
# -----------------------------------------------

def load_nn_model(ckpt_path, device):
    """Loads the neural network from checkpoint."""
    print(f"Loading model from: {ckpt_path}")
    # We use strict=False because sometimes lighting modules have extra keys
    model = NSModel.load_from_checkpoint(ckpt_path, strict=False).eval().to(device)
    return model

def get_nn_jacobian_svd(u_input, model, dim, k=100):
    """
    Computes SVD of the Neural Network Jacobian using Autograd.
    """
    device = next(model.parameters()).device
    
    # Prepare input tensor (Batch=1, Channel=1, H, W)
    if not torch.is_tensor(u_input):
        u_tensor = torch.tensor(u_input, dtype=torch.float32, device=device)
    else:
        u_tensor = u_input.to(device)

    # Ensure shape (1, 1, DIM, DIM)
    if u_tensor.ndim == 2: u_tensor = u_tensor.unsqueeze(0).unsqueeze(0)
    elif u_tensor.ndim == 3: u_tensor = u_tensor.unsqueeze(0)
    
    # We must enable gradients for the input to compute J^T v
    u_tensor.requires_grad_(True)
    
    # Precompute output shape
    # NN output is usually (Batch, Channel, H, W)
    y_pred = model(u_tensor) 
    
    # ---------------------------------------------------------
    # Helper: Forward Mode (Jv) using Autograd
    # ---------------------------------------------------------
    def mv(v):
        """ J @ v """
        v_tensor = torch.tensor(v.reshape(u_tensor.shape), dtype=torch.float32, device=device)
        
        # torch.autograd.functional.jvp calculates J * v efficiently
        # We wrap it to handle the 'functional' inputs
        def forward_fn(x):
            return model(x)
        
        # JVP returns (output, jvp_out)
        _, jvp_out = torch.autograd.functional.jvp(forward_fn, u_tensor, v=v_tensor)
        
        return jvp_out.detach().cpu().numpy().reshape(-1)

    # ---------------------------------------------------------
    # Helper: Adjoint Mode (J^T v) using Autograd
    # ---------------------------------------------------------
    def rmv(v):
        """ J.T @ v """
        # We need to clear grads before backward pass
        if u_tensor.grad is not None:
            u_tensor.grad.zero_()
            
        v_tensor = torch.tensor(v.reshape(y_pred.shape), dtype=torch.float32, device=device)
        
        # Standard VJP: backward() with grad_tensors
        # Since y_pred was computed from u_tensor, the graph is connected.
        # However, we need to re-run forward if we don't retain graph, 
        # but torch.autograd.grad handles this if we provide inputs/outputs.
        
        # Simpler approach: use autograd.grad
        # Note: We must ensure y_pred graph is alive. 
        # For iterative solvers, we often re-forward or retain_graph=True.
        # Ideally, we define a fresh computation graph for stability.
        
        out = model(u_tensor)
        grad = torch.autograd.grad(
            outputs=out,
            inputs=u_tensor,
            grad_outputs=v_tensor,
            create_graph=False,
            retain_graph=False 
        )[0]
        
        return grad.detach().cpu().numpy().reshape(-1)

    # Define Linear Operator
    n_params = dim * dim
    J_op = LinearOperator((n_params, n_params), matvec=mv, rmatvec=rmv)

    print(f"   Computing top {k} SVs (NN Autograd)...")
    try:
        _, s, _ = svds(J_op, k=k)
        return s[::-1]
    except Exception as e:
        print(f"   SVD failed: {e}")
        return np.zeros(k)

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Settings for the two experiments
    if TYPE == "MSE":
        experiments = [
            {"name": "Darcy_mse",
                "ckpt": CKPT_DARCY_MSE,
                "dim": DIM_DARCY,
                "tau": 3.0,
                "alpha": 2.0},
            {"name": "NS_mse",
                "ckpt": CKPT_NS_MSE,
                "dim": DIM_NS,
                "tau": 3.0,
                "alpha": 2.5}]    
    else:
        experiments = [
            {"name": "Darcy",
                "ckpt": CKPT_DARCY,
                "dim": DIM_DARCY,
                "tau": 3.0,
                "alpha": 2.0},]
            # {"name": "NS",
            #     "ckpt": CKPT_NS,
            #     "dim": DIM_NS,
            #     "tau": 3.0,
            #     "alpha": 2.5}]

    for exp in experiments:
        name = exp["name"]
        dim  = exp["dim"]
        print(f"\n================ STARTING {name} (Dim: {dim}) ================")
        
        # 1. Load Neural Network
        try:
            model = load_nn_model(exp["ckpt"], device)
        except Exception as e:
            print(f"Skipping {name} due to loading error (check path): {e}")
            continue

        # 2. Setup Random Field for Inputs
        grf = GaussianRandomField(2, dim, alpha=exp["alpha"], tau=exp["tau"])
        
        results = {}
        plt.figure(figsize=(10, 6))

        # 3. Process Samples
        for i in range(NUM_SAMPLES):
            print(f"--- {name} Sample {i+1}/{NUM_SAMPLES} ---")
            
            # Generate input
            u_sample = grf.sample(1)
            u_np = u_sample[0].astype(np.float32)

            # Compute SVD
            s_values = get_nn_jacobian_svd(u_np, model, dim, k=K_SINGULAR_VALUES)
            
            results[f'Sample_{i+1}'] = s_values
            plt.plot(s_values, label=f'Sample {i+1}', linewidth=2, alpha=0.8)

        # 4. Save CSV
        csv_name = f'plot/svd_nn_{name}.csv'
        pd.DataFrame(results).to_csv(csv_name, index_label="Rank")
        print(f"Saved {csv_name}")

        # 5. Plot
        plt.yscale('log')
        plt.title(f'NN Jacobian SVD Decay: {name} ({dim}x{dim})')
        plt.ylabel(r'Singular Value ($\sigma_i$)')
        plt.xlabel(r'Index $i$')
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'plot/svd_nn_decay_{name}.png')
        print(f"Saved plot plot/svd_nn_decay_{name}.png")