#!/usr/bin/env python3
"""
Minimal test to isolate device issue
"""

import torch
import torch.nn as nn

# Test basic device handling
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create simple tensors
print("\n1. Testing basic tensor creation...")
x = torch.randn(5, 2, device=device)
print(f"Input tensor device: {x.device}")

# Create simple network
print("\n2. Testing network creation and device placement...")
net = nn.Sequential(
    nn.Linear(2, 10),
    nn.Tanh(),
    nn.Linear(10, 1)
).to(device)

print(f"Network device: {next(net.parameters()).device}")

# Test forward pass
print("\n3. Testing forward pass...")
try:
    y = net(x)
    print(f"✅ Forward pass successful!")
    print(f"Output device: {y.device}")
    print(f"Output shape: {y.shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")

# Now test our implementation
print("\n4. Testing our Laminar implementation...")
try:
    from laminar_pytorch import LaminarPyTorch
    
    # Create with very small size
    laminar = LaminarPyTorch(unit=0.1, nx=5, ny=4, nu=4.0e-2, stokes=True, device=device)
    
    print(f"Laminar created successfully")
    print(f"Device attribute: {laminar.device}")
    print(f"Coords device: {laminar.coords.device}")
    print(f"Network device: {next(laminar.velocity_net.parameters()).device}")
    
    # Test coordinate processing
    coords_test = laminar.coords[:10]  # Just first 10 points
    print(f"Coords test device: {coords_test.device}")
    
    # Test network forward
    velocity_test = laminar.velocity_net(coords_test)
    print(f"✅ Network forward successful!")
    print(f"Velocity output device: {velocity_test.device}")
    
except Exception as e:
    print(f"❌ Laminar test failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed.")