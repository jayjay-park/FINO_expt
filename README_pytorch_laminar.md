# PyTorch Implementation of Laminar Jet PDE Model

This repository contains PyTorch implementations of the Laminar Jet PDE model, converted from the original FEniCS implementation to be fully differentiable using automatic differentiation.

## Overview

The original code uses FEniCS finite element method with adjoint methods to compute gradients. These PyTorch implementations provide the same functionality but with automatic differentiation, making the entire pipeline differentiable and suitable for machine learning applications.

## Files

### 1. `laminar_pytorch.py`
A direct PyTorch conversion using finite difference operators and neural networks for PDE approximation.

**Features:**
- Finite difference discretization of the domain
- Neural network approximation of the PDE solution
- Automatic differentiation for gradient computation
- Compatible API with the original FEniCS implementation

### 2. `laminar_pytorch_pinn.py` (Recommended)
A Physics-Informed Neural Network (PINN) implementation that directly solves the Navier-Stokes equations.

**Features:**
- Physics-Informed Neural Networks for PDE solving
- Automatic enforcement of boundary conditions
- Direct optimization of PDE residuals
- Visualization capabilities
- More accurate and stable than the finite difference approach

## Key Differences from Original FEniCS Code

| Aspect | Original FEniCS | PyTorch Implementation |
|--------|----------------|----------------------|
| **PDE Solving** | Finite Element Method | Neural Networks (PINN) |
| **Gradient Computation** | Adjoint Methods | Automatic Differentiation |
| **Mesh** | Unstructured FEM mesh | Structured grid or continuous domain |
| **Boundary Conditions** | Dirichlet/Neumann BCs | Soft constraint in loss function |
| **Differentiability** | Through adjoint equations | Native PyTorch autograd |
| **Performance** | Fast for single solve | Slower per solve, but fully differentiable |

## Usage

### Basic Usage

```python
import torch
from laminar_pytorch_pinn import LaminarPINN

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create model
laminar = LaminarPINN(
    unit=0.1,           # Length scale
    nx=30, ny=25,       # Grid resolution
    nu=4.0e-2,          # Viscosity
    stokes=True,        # Use Stokes equations (linear)
    device=device
)

# Generate random parameters
dim_theta = 5
theta = torch.randn(dim_theta, device=device)

# Get inflow profile (Karhunen-Loeve expansion)
inflow_profile = laminar.inflow_profile(theta, sigma=0.1, alpha=1.0, s=0.6)

# Solve forward problem
u, v, p = laminar.solve_forward(inflow_profile, train_epochs=1000)

# Get observations at outlet
obs, idx, loc = laminar.get_obs(inflow_profile)

# Create data misfit functional
misfit = laminar.data_misfit(obs, prec=100.0, idx=idx, loc=loc)

# Compute misfit and gradient using automatic differentiation
nll, dnll = laminar.get_geom(theta, misfit, sigma=0.1, alpha=1.0, s=0.6)

print(f"Negative log-likelihood: {nll.item()}")
print(f"Gradient: {dnll.detach().cpu().numpy()}")
```

### Advanced Usage: Optimization Loop

```python
import torch.optim as optim

# Initialize parameters
theta = torch.randn(dim_theta, device=device, requires_grad=True)
optimizer = optim.Adam([theta], lr=0.01)

# Optimization loop
for epoch in range(100):
    optimizer.zero_grad()
    
    # Compute misfit
    inflow_profile = laminar.inflow_profile(theta, sigma=0.1, alpha=1.0, s=0.6)
    u_outlet, _, _ = laminar.get_solution(laminar.outlet_points)
    loss = misfit.eval(u_outlet)
    
    # Backpropagate
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
```

## Mathematical Formulation

### Governing Equations

The code solves the incompressible Navier-Stokes equations:

**Momentum equations:**
```
∂u/∂t + u·∇u = -∇p + ν∇²u + f
```

**Continuity equation:**
```
∇·u = 0
```

For the Stokes case (`stokes=True`), the nonlinear advection term `u·∇u` is omitted.

### Boundary Conditions

- **Inlet (x=0):** Prescribed velocity profile from Karhunen-Loeve expansion
- **Outlet (x=Lx):** Natural boundary conditions (∂u/∂n = 0)
- **Walls (y=±Ly/2):** No-slip conditions (u = v = 0)

### Karhunen-Loeve Expansion

The inflow profile is parameterized using a KL expansion:

```
u_inlet(y) = σ Σ θᵢ λᵢ^(-s/2) cos(πi·y)
```

where:
- `θᵢ` are the KL coefficients (parameters to be optimized)
- `λᵢ = α + (πi)²` are the eigenvalues
- `σ`, `α`, `s` are hyperparameters

## Installation Requirements

```bash
pip install torch torchvision
pip install numpy matplotlib
pip install scipy  # Optional, for comparisons
```

## Performance Considerations

### PINN Training
- **Training time:** Each PINN solve requires 500-2000 training epochs
- **Memory usage:** Scales with network size and number of collocation points
- **Accuracy:** Depends on network architecture and training convergence

### Recommendations
- Use GPU acceleration when available
- Start with Stokes equations (`stokes=True`) for faster convergence
- Reduce `train_epochs` for faster but less accurate solutions
- Use smaller networks for prototyping, larger for production

## Comparison with Original FEniCS Implementation

### Advantages of PyTorch Version
1. **Full differentiability:** No need for adjoint equation derivation
2. **Flexibility:** Easy to modify physics, boundary conditions, or objectives
3. **Integration:** Natural integration with ML pipelines
4. **Extensibility:** Easy to add new physics or neural network architectures

### Disadvantages
1. **Speed:** Slower than optimized FEM solvers for single evaluations
2. **Accuracy:** May require careful tuning of network architecture and training
3. **Memory:** Higher memory usage due to computational graph storage

## Examples and Testing

Run the test function to verify the implementation:

```python
# Test the PINN implementation
python laminar_pytorch_pinn.py
```

This will:
1. Generate random KL coefficients
2. Solve the forward problem
3. Compute synthetic observations
4. Calculate gradients using automatic differentiation
5. Optionally compare with finite differences
6. Plot the solution fields

## Extending the Code

### Adding New Physics
To add new physical terms, modify the `compute_pde_residual` method:

```python
def compute_pde_residual(self, x: torch.Tensor) -> torch.Tensor:
    # ... existing code ...
    
    # Add new physics term
    new_term = self.compute_new_physics_term(u, v, p, x)
    momentum_x += new_term
    
    return momentum_x, momentum_y, continuity
```

### Changing Boundary Conditions
Modify the `compute_boundary_loss` method to implement different boundary conditions:

```python
def compute_boundary_loss(self, inflow_profile: torch.Tensor) -> torch.Tensor:
    # ... existing code ...
    
    # Add new boundary condition
    new_bc_loss = self.compute_new_boundary_condition()
    total_bc_loss += new_bc_loss
    
    return total_bc_loss
```

### Custom Neural Network Architectures
Modify the network architecture in `__init__`:

```python
# Example: Add residual connections
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        return x + self.net(x)

# Use in main network
self.layers = nn.ModuleList([
    nn.Linear(2, 128),
    ResidualBlock(128),
    ResidualBlock(128),
    nn.Linear(128, 3)
])
```

## Troubleshooting

### Common Issues

1. **Training doesn't converge:**
   - Increase `train_epochs`
   - Adjust learning rate
   - Check boundary condition weights (`bc_weight`)
   - Try different network architectures

2. **Gradients are zero or very small:**
   - Ensure `theta.requires_grad_(True)`
   - Check that the loss function depends on theta
   - Verify the computational graph is not broken

3. **Memory errors:**
   - Reduce network size or number of collocation points
   - Use gradient checkpointing
   - Process in smaller batches

4. **Slow performance:**
   - Use GPU acceleration
   - Reduce `train_epochs` for prototyping
   - Consider using mixed precision training

### Debugging Tips

1. **Visualize the solution:** Use `plot_solution()` to check if the PDE is being solved correctly
2. **Check residuals:** Monitor PDE and boundary condition losses during training
3. **Gradient checking:** Use finite differences to verify automatic differentiation
4. **Start simple:** Begin with Stokes equations and simple geometries

## Contributing

To contribute to this implementation:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure backward compatibility with the original API
5. Submit a pull request

## License

This code is derived from the original FEniCS implementation and maintains the same license terms.

## References

1. Original FEniCS implementation by Shiwei Lan
2. Physics-Informed Neural Networks: [Raissi et al., 2019](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
3. PyTorch documentation: https://pytorch.org/docs/

## Citation

If you use this code in your research, please cite both the original FEniCS implementation and this PyTorch conversion.