#!/usr/bin/env python3
"""
Test script to verify device handling in PyTorch Laminar implementations
"""

import torch

def test_laminar_pytorch():
    """Test the basic laminar_pytorch.py implementation"""
    try:
        from laminar_pytorch import LaminarPyTorch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Testing LaminarPyTorch on device: {device}")
        
        # Create model with smaller size for quick testing
        laminar = LaminarPyTorch(unit=0.1, nx=10, ny=8, nu=4.0e-2, stokes=True, device=device)
        
        # Test device placement
        print(f"Model device: {next(laminar.velocity_net.parameters()).device}")
        print(f"Coords device: {laminar.coords.device}")
        print(f"Operators device: {laminar.grad_x_op.device}")
        
        # Generate simple theta
        theta = torch.randn(3, device=device)
        print(f"Theta device: {theta.device}")
        
        # Test inflow profile generation
        inflow_profile = laminar.inflow_profile(theta, sigma=0.1, alpha=1.0, s=0.6)
        print(f"Inflow profile device: {inflow_profile.device}")
        print(f"Inflow profile shape: {inflow_profile.shape}")
        
        # Test forward solve (this was failing before)
        print("Testing forward solve...")
        u, v, p, l = laminar.solve_forward(inflow_profile)
        print(f"Solution shapes: u={u.shape}, v={v.shape}, p={p.shape}, l={l.shape}")
        print(f"Solution devices: u={u.device}, v={v.device}, p={p.device}, l={l.device}")
        
        print("✅ LaminarPyTorch test passed!")
        return True
        
    except Exception as e:
        print(f"❌ LaminarPyTorch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_laminar_pinn():
    """Test the PINN implementation"""
    try:
        from laminar_pytorch_pinn import LaminarPINN
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nTesting LaminarPINN on device: {device}")
        
        # Create model with smaller size for quick testing
        laminar = LaminarPINN(unit=0.1, nx=15, ny=12, nu=4.0e-2, stokes=True, device=device)
        
        # Test device placement
        print(f"Model device: {next(laminar.parameters()).device}")
        print(f"Interior points device: {laminar.interior_points.device}")
        print(f"Inlet points device: {laminar.inlet_points.device}")
        
        # Generate simple theta
        theta = torch.randn(3, device=device)
        print(f"Theta device: {theta.device}")
        
        # Test inflow profile generation
        inflow_profile = laminar.inflow_profile(theta, sigma=0.1, alpha=1.0, s=0.6)
        print(f"Inflow profile device: {inflow_profile.device}")
        print(f"Inflow profile shape: {inflow_profile.shape}")
        
        # Test forward pass through network
        print("Testing network forward pass...")
        test_points = laminar.inlet_points[:5]  # Just test a few points
        u, v, p = laminar.get_solution(test_points)
        print(f"Network output shapes: u={u.shape}, v={v.shape}, p={p.shape}")
        print(f"Network output devices: u={u.device}, v={v.device}, p={p.device}")
        
        print("✅ LaminarPINN test passed!")
        return True
        
    except Exception as e:
        print(f"❌ LaminarPINN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run device tests"""
    print("=" * 60)
    print("Testing Device Handling in PyTorch Laminar Implementations")
    print("=" * 60)
    
    # Check PyTorch and CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run tests
    test1_passed = test_laminar_pytorch()
    test2_passed = test_laminar_pinn()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 All device tests passed!")
        print("The implementations should now work correctly on both CPU and GPU.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    print("=" * 60)
    
    return test1_passed and test2_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)