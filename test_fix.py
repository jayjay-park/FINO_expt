#!/usr/bin/env python3
"""
Test script to verify the device fix
"""

import torch

def test_device_fix():
    """Test if the device fix works"""
    print("Testing device fix...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        from laminar_pytorch import LaminarPyTorch
        
        # Create model (small size for quick test)
        print("Creating LaminarPyTorch model...")
        laminar = LaminarPyTorch(unit=0.1, nx=8, ny=6, nu=4.0e-2, stokes=True, device=device)
        
        # Check device placement
        print("Checking device placement...")
        print(f"  Networks on {next(laminar.velocity_net.parameters()).device}")
        print(f"  Coordinates on {laminar.coords.device}")
        
        # Test theta
        print("Creating test parameters...")
        theta = torch.randn(3, device=device)
        
        # Test inflow profile
        print("Testing inflow profile...")
        inflow_profile = laminar.inflow_profile(theta, sigma=0.1, alpha=1.0, s=0.6)
        print(f"  Inflow profile created on {inflow_profile.device}")
        
        # Test forward solve (this was the failing step)
        print("Testing forward solve...")
        u, v, p, l = laminar.solve_forward(inflow_profile)
        print(f"  ✅ Forward solve successful!")
        print(f"  Solution shapes: u={u.shape}, v={v.shape}, p={p.shape}, l={l.shape}")
        
        # Test observations
        print("Testing observations...")
        obs, idx, loc = laminar.get_obs(inflow_profile)
        print(f"  ✅ Observations obtained: {len(obs)} points")
        
        print("\n🎉 All tests passed! The device fix is working.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_device_fix()
    if success:
        print("\nYou can now run the original script:")
        print("python laminar_pytorch.py")
    else:
        print("\nThere are still issues. Please check the error above.")