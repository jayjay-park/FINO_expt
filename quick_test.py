#!/usr/bin/env python3
"""
Quick test to verify the device fix
"""

import torch
from laminar_pytorch import LaminarPyTorch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    laminar = LaminarPyTorch(unit=0.1, nx=10, ny=8, nu=4.0e-2, stokes=True, device=device)
    
    # Check devices
    print(f"Networks on device: {next(laminar.velocity_net.parameters()).device}")
    print(f"Coordinates on device: {laminar.coords.device}")
    
    # Test with simple parameters
    theta = torch.randn(3, device=device)
    inflow_profile = laminar.inflow_profile(theta, sigma=0.1, alpha=1.0, s=0.6)
    
    print(f"Inflow profile shape: {inflow_profile.shape}")
    print(f"Inflow profile device: {inflow_profile.device}")
    
    # This should now work without device errors
    try:
        u, v, p, l = laminar.solve_forward(inflow_profile)
        print("✅ Forward solve successful!")
        print(f"Solution shapes: u={u.shape}, v={v.shape}, p={p.shape}, l={l.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    print("Test", "PASSED" if success else "FAILED")