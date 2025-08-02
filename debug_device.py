#!/usr/bin/env python3
"""
Debug script to identify device mismatches in the Laminar implementation
"""

import torch
from laminar_pytorch import LaminarPyTorch

def debug_device_placement():
    """Debug device placement step by step"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Target device: {device}")
    
    # Create model
    print("\n1. Creating model...")
    laminar = LaminarPyTorch(unit=0.1, nx=10, ny=8, nu=4.0e-2, stokes=True, device=device)
    
    # Check all component devices
    print("\n2. Checking component devices...")
    print(f"  velocity_net device: {next(laminar.velocity_net.parameters()).device}")
    print(f"  pressure_net device: {next(laminar.pressure_net.parameters()).device}")
    print(f"  lagrange_net device: {next(laminar.lagrange_net.parameters()).device}")
    print(f"  coords device: {laminar.coords.device}")
    print(f"  X device: {laminar.X.device}")
    print(f"  Y device: {laminar.Y.device}")
    print(f"  inlet_y_coords device: {laminar.inlet_y_coords.device}")
    print(f"  inlet_mask device: {laminar.inlet_mask.device}")
    print(f"  outlet_mask device: {laminar.outlet_mask.device}")
    print(f"  bounding_mask device: {laminar.bounding_mask.device}")
    print(f"  grad_x_op device: {laminar.grad_x_op.device}")
    print(f"  grad_y_op device: {laminar.grad_y_op.device}")
    print(f"  laplacian_op device: {laminar.laplacian_op.device}")
    
    # Test theta creation
    print("\n3. Creating theta...")
    theta = torch.randn(3, device=device)
    print(f"  theta device: {theta.device}")
    
    # Test inflow profile
    print("\n4. Creating inflow profile...")
    inflow_profile = laminar.inflow_profile(theta, sigma=0.1, alpha=1.0, s=0.6)
    print(f"  inflow_profile device: {inflow_profile.device}")
    print(f"  inflow_profile shape: {inflow_profile.shape}")
    
    # Test coordinate normalization step by step
    print("\n5. Testing coordinate normalization...")
    coords_orig = laminar.coords
    print(f"  coords_orig device: {coords_orig.device}")
    
    coords_clone = coords_orig.clone()
    print(f"  coords_clone device: {coords_clone.device}")
    
    coords_norm = coords_clone
    coords_norm[:, 0] /= laminar.Lx
    coords_norm[:, 1] /= (0.5 * laminar.Ly)
    print(f"  coords_norm device: {coords_norm.device}")
    print(f"  coords_norm shape: {coords_norm.shape}")
    
    # Test neural network forward pass
    print("\n6. Testing neural network forward pass...")
    try:
        velocity = laminar.velocity_net(coords_norm)
        print(f"  ✅ velocity_net forward pass successful")
        print(f"  velocity device: {velocity.device}")
        print(f"  velocity shape: {velocity.shape}")
    except Exception as e:
        print(f"  ❌ velocity_net forward pass failed: {e}")
        return False
    
    try:
        pressure = laminar.pressure_net(coords_norm).squeeze()
        print(f"  ✅ pressure_net forward pass successful")
        print(f"  pressure device: {pressure.device}")
        print(f"  pressure shape: {pressure.shape}")
    except Exception as e:
        print(f"  ❌ pressure_net forward pass failed: {e}")
        return False
    
    try:
        lagrange = laminar.lagrange_net(coords_norm).squeeze()
        print(f"  ✅ lagrange_net forward pass successful")
        print(f"  lagrange device: {lagrange.device}")
        print(f"  lagrange shape: {lagrange.shape}")
    except Exception as e:
        print(f"  ❌ lagrange_net forward pass failed: {e}")
        return False
    
    print("\n7. All device checks passed! ✅")
    return True

if __name__ == "__main__":
    success = debug_device_placement()
    if success:
        print("\n🎉 Device placement is correct!")
    else:
        print("\n⚠️ Device placement issues found.")