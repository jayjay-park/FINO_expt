#!/usr/bin/env python
"""
PyTorch implementation of Laminar-Jet PDE model
Converted from FEniCS implementation to be fully differentiable
-----------------------------------
The purpose of this script is to obtain geometric quantities, misfit, its gradient and the associated metric 
using automatic differentiation in PyTorch instead of adjoint methods.
--To run demo:                     python laminar_pytorch.py
--To initialize problem:     e.g.  laminar=LaminarPyTorch(args); inflow=laminar.inflow_profile(args)
--To obtain observations:          obs,idx,loc=laminar.get_obs(inflow)
--To define data misfit class:     misfit=laminar.data_misfit(args)
--To obtain geometric quantities:  nll,dnll = laminar.get_geom # misfit value and gradient
-----------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import warnings

class LaminarPyTorch(nn.Module):
    """
    PyTorch implementation of the Laminar Jet PDE model.
    Uses neural network-based PDE solving with automatic differentiation.
    """
    
    def __init__(self, 
                 unit: float = 1.0,
                 nx: int = 40, 
                 ny: int = 30,
                 nu: float = 1.0e-1,
                 beta: float = 0.5,
                 stokes: bool = False,
                 nugg: float = 1.0e-20,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        super(LaminarPyTorch, self).__init__()
        
        # Geometry parameters
        self.unit = unit
        self.Lx = 10 * self.unit
        self.Ly = 8 * self.unit
        self.nx = nx
        self.ny = ny
        
        # Physical parameters
        self.nu = nu
        self.beta = beta
        self.stokes = stokes
        self.nugg = nugg
        
        # Device and dtype
        self.device = device
        self.dtype = dtype
        
        # Boundary labels
        self.INLET = 1
        self.OUTLET = 2
        self.BOUNDING = 3
        
        # Initialize mesh and finite difference operators
        self.setup_mesh()
        self.setup_operators()
        
        # Neural network for PDE solution
        self.setup_neural_solver()
        
        # Solution count for tracking
        self.soln_count = torch.zeros(6, device=device, dtype=dtype)
        
        # Ensure all components are on correct device (call after everything is set up)
        self.to_device()
        
    def setup_mesh(self):
        """Setup the computational mesh using finite differences."""
        # Create coordinate grids
        x = torch.linspace(0, self.Lx, self.nx + 1, device=self.device, dtype=self.dtype)
        y = torch.linspace(-0.5 * self.Ly, 0.5 * self.Ly, self.ny + 1, device=self.device, dtype=self.dtype)
        
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        
        # Create mesh grids
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
        
        # Flatten for easier manipulation
        self.coords = torch.stack([self.X.flatten(), self.Y.flatten()], dim=1)
        
        # Boundary masks
        self.inlet_mask = (self.X <= self.dx/2).flatten()
        self.outlet_mask = (self.X >= self.Lx - self.dx/2).flatten()
        self.bounding_mask = ((torch.abs(self.Y) >= 0.5 * self.Ly - self.dy/2)).flatten()
        
        # Inlet boundary coordinates for inflow profile
        inlet_coords = self.coords[self.inlet_mask]
        self.inlet_y_coords = inlet_coords[:, 1]
    
    def to_device(self):
        """Ensure all tensors are moved to the correct device."""
        # Move coordinate tensors (check if they exist first)
        if hasattr(self, 'coords'):
            self.coords = self.coords.to(self.device)
        if hasattr(self, 'X'):
            self.X = self.X.to(self.device)
        if hasattr(self, 'Y'):
            self.Y = self.Y.to(self.device)
        if hasattr(self, 'inlet_y_coords'):
            self.inlet_y_coords = self.inlet_y_coords.to(self.device)
        
        # Move boundary masks
        if hasattr(self, 'inlet_mask'):
            self.inlet_mask = self.inlet_mask.to(self.device)
        if hasattr(self, 'outlet_mask'):
            self.outlet_mask = self.outlet_mask.to(self.device)
        if hasattr(self, 'bounding_mask'):
            self.bounding_mask = self.bounding_mask.to(self.device)
        
        # Move operators
        if hasattr(self, 'grad_x_op'):
            self.grad_x_op = self.grad_x_op.to(self.device)
        if hasattr(self, 'grad_y_op'):
            self.grad_y_op = self.grad_y_op.to(self.device)
        if hasattr(self, 'laplacian_op'):
            self.laplacian_op = self.laplacian_op.to(self.device)
        
        # Move solution count
        if hasattr(self, 'soln_count'):
            self.soln_count = self.soln_count.to(self.device)
    
    def to(self, device):
        """Override to method to handle custom tensors."""
        super().to(device)
        self.device = device
        self.to_device()
        return self
        
    def setup_operators(self):
        """Setup finite difference operators for gradients and Laplacian."""
        # Create finite difference matrices for derivatives
        nx, ny = self.nx + 1, self.ny + 1
        n_total = nx * ny
        
        # Helper function to convert 2D indices to 1D
        def idx_2d_to_1d(i, j):
            return i * ny + j
        
        # Gradient operators (using central differences where possible)
        self.grad_x_op = torch.zeros(n_total, n_total, device=self.device, dtype=self.dtype)
        self.grad_y_op = torch.zeros(n_total, n_total, device=self.device, dtype=self.dtype)
        self.laplacian_op = torch.zeros(n_total, n_total, device=self.device, dtype=self.dtype)
        
        for i in range(nx):
            for j in range(ny):
                idx = idx_2d_to_1d(i, j)
                
                # x-derivative
                if i == 0:  # Forward difference at left boundary
                    self.grad_x_op[idx, idx_2d_to_1d(i+1, j)] = 1.0 / self.dx
                    self.grad_x_op[idx, idx] = -1.0 / self.dx
                elif i == nx - 1:  # Backward difference at right boundary
                    self.grad_x_op[idx, idx] = 1.0 / self.dx
                    self.grad_x_op[idx, idx_2d_to_1d(i-1, j)] = -1.0 / self.dx
                else:  # Central difference
                    self.grad_x_op[idx, idx_2d_to_1d(i+1, j)] = 0.5 / self.dx
                    self.grad_x_op[idx, idx_2d_to_1d(i-1, j)] = -0.5 / self.dx
                
                # y-derivative
                if j == 0:  # Forward difference at bottom boundary
                    self.grad_y_op[idx, idx_2d_to_1d(i, j+1)] = 1.0 / self.dy
                    self.grad_y_op[idx, idx] = -1.0 / self.dy
                elif j == ny - 1:  # Backward difference at top boundary
                    self.grad_y_op[idx, idx] = 1.0 / self.dy
                    self.grad_y_op[idx, idx_2d_to_1d(i, j-1)] = -1.0 / self.dy
                else:  # Central difference
                    self.grad_y_op[idx, idx_2d_to_1d(i, j+1)] = 0.5 / self.dy
                    self.grad_y_op[idx, idx_2d_to_1d(i, j-1)] = -0.5 / self.dy
                
                # Laplacian (5-point stencil)
                if 0 < i < nx - 1 and 0 < j < ny - 1:
                    self.laplacian_op[idx, idx] = -2.0 / (self.dx**2) - 2.0 / (self.dy**2)
                    self.laplacian_op[idx, idx_2d_to_1d(i+1, j)] = 1.0 / (self.dx**2)
                    self.laplacian_op[idx, idx_2d_to_1d(i-1, j)] = 1.0 / (self.dx**2)
                    self.laplacian_op[idx, idx_2d_to_1d(i, j+1)] = 1.0 / (self.dy**2)
                    self.laplacian_op[idx, idx_2d_to_1d(i, j-1)] = 1.0 / (self.dy**2)
    
    def setup_neural_solver(self):
        """Setup neural network for PDE solution approximation."""
        n_points = (self.nx + 1) * (self.ny + 1)
        
        # Neural network for velocity field (u, v)
        self.velocity_net = nn.Sequential(
            nn.Linear(2, 128),  # Input: (x, y)
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 2)   # Output: (u, v)
        ).to(self.device)
        
        # Neural network for pressure field
        self.pressure_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)    # Output: pressure
        ).to(self.device)
        
        # Neural network for Lagrange multiplier
        self.lagrange_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)    # Output: Lagrange multiplier
        ).to(self.device)
    
    def inflow_profile(self, theta: torch.Tensor, sigma: float = 1.2, 
                      alpha: float = 0.1, s: float = 0.6) -> torch.Tensor:
        """
        Karhunen-Loeve expansion of inflow profile.
        
        Args:
            theta: KL coefficients
            sigma, alpha, s: KL expansion parameters
            
        Returns:
            Inflow velocity profile at inlet boundary
        """
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, device=self.device, dtype=self.dtype)
        
        l = len(theta)
        seq = torch.arange(l, device=self.device, dtype=self.dtype)
        
        # KL expansion
        eigenvals = (alpha + (np.pi * seq)**2)**(-s/2)
        cos_terms = torch.cos(np.pi * seq.unsqueeze(1) * self.inlet_y_coords.unsqueeze(0))
        
        profile = sigma * torch.sum(theta.unsqueeze(1) * eigenvals.unsqueeze(1) * cos_terms, dim=0)
        
        return profile
    
    def apply_boundary_conditions(self, u: torch.Tensor, v: torch.Tensor, 
                                 p: torch.Tensor, l: torch.Tensor,
                                 inflow_profile: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Apply boundary conditions to the solution."""
        # Apply boundary conditions
        u_bc = u.clone()
        v_bc = v.clone()
        p_bc = p.clone()
        l_bc = l.clone()
        
        # Inlet: u = inflow_profile, v = 0
        u_bc[self.inlet_mask] = inflow_profile
        v_bc[self.inlet_mask] = 0.0
        
        # Bounding walls: v = 0
        v_bc[self.bounding_mask] = 0.0
        
        # Outlet: natural boundary conditions (handled in PDE residual)
        
        return u_bc, v_bc, p_bc, l_bc
    
    def compute_pde_residual(self, u: torch.Tensor, v: torch.Tensor, 
                           p: torch.Tensor, l: torch.Tensor,
                           inflow_profile: torch.Tensor) -> torch.Tensor:
        """
        Compute the PDE residual for the Navier-Stokes equations.
        """
        # Apply boundary conditions
        u, v, p, l = self.apply_boundary_conditions(u, v, p, l, inflow_profile)
        
        # Compute gradients
        u_x = torch.matmul(self.grad_x_op, u)
        u_y = torch.matmul(self.grad_y_op, u)
        v_x = torch.matmul(self.grad_x_op, v)
        v_y = torch.matmul(self.grad_y_op, v)
        p_x = torch.matmul(self.grad_x_op, p)
        p_y = torch.matmul(self.grad_y_op, p)
        
        # Compute Laplacians
        u_xx_yy = torch.matmul(self.laplacian_op, u)
        v_xx_yy = torch.matmul(self.laplacian_op, v)
        
        # Momentum equations
        if self.stokes:
            # Linear Stokes equations
            momentum_x = self.nu * u_xx_yy - p_x
            momentum_y = self.nu * v_xx_yy - p_y
        else:
            # Nonlinear Navier-Stokes equations
            momentum_x = self.nu * u_xx_yy - u * u_x - v * u_y - p_x
            momentum_y = self.nu * v_xx_yy - u * v_x - v * v_y - p_y
        
        # Continuity equation
        continuity = u_x + v_y
        
        # Lagrange multiplier equation (inlet condition)
        lagrange_eq = torch.zeros_like(l)
        lagrange_eq[self.inlet_mask] = u[self.inlet_mask] - inflow_profile
        
        # Combine residuals
        residual = torch.cat([momentum_x, momentum_y, continuity, lagrange_eq])
        
        return residual
    
    def solve_forward(self, inflow_profile: torch.Tensor, 
                     max_iter: int = 1000, tol: float = 1e-6) -> Tuple[torch.Tensor, ...]:
        """
        Solve the forward PDE using the neural network approach.
        """
        # Normalize coordinates for neural network input
        coords_norm = self.coords.clone()
        coords_norm[:, 0] /= self.Lx
        coords_norm[:, 1] /= (0.5 * self.Ly)
        
        # Get initial solution from neural networks
        velocity = self.velocity_net(coords_norm)
        pressure = self.pressure_net(coords_norm).squeeze()
        lagrange = self.lagrange_net(coords_norm).squeeze()
        
        u, v = velocity[:, 0], velocity[:, 1]
        
        # Apply boundary conditions
        u, v, pressure, lagrange = self.apply_boundary_conditions(u, v, pressure, lagrange, inflow_profile)
        
        self.soln_count[2] += 1  # Forward solve count
        
        return u, v, pressure, lagrange
    
    def get_obs(self, inflow_profile: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get observations at outlet boundary.
        """
        # Solve forward problem
        u, v, p, l = self.solve_forward(inflow_profile)
        
        # Extract observations at outlet
        outlet_indices = torch.where(self.outlet_mask)[0]
        outlet_coords = self.coords[outlet_indices]
        obs = u[outlet_indices]
        
        return obs, outlet_indices, outlet_coords
    
    class DataMisfit:
        """Data misfit class for computing observation-based loss."""
        
        def __init__(self, outer_obj, obs: torch.Tensor, prec: float, 
                     idx: Optional[torch.Tensor] = None, 
                     loc: Optional[torch.Tensor] = None):
            self.outer_obj = outer_obj
            self.obs = obs if isinstance(obs, torch.Tensor) else torch.tensor(obs, device=outer_obj.device, dtype=outer_obj.dtype)
            self.prec = prec
            self.idx = idx
            self.loc = loc
        
        def eval(self, u: torch.Tensor) -> torch.Tensor:
            """Evaluate data misfit."""
            if self.idx is not None:
                u_obs = u[self.idx]
            else:
                u_obs = u  # Assume u is already at observation points
            
            diff = u_obs - self.obs
            misfit = 0.5 * self.prec * torch.sum(diff**2)
            return misfit
    
    def data_misfit(self, obs: torch.Tensor, prec: float, 
                   idx: Optional[torch.Tensor] = None, 
                   loc: Optional[torch.Tensor] = None) -> DataMisfit:
        """Create data misfit object."""
        return self.DataMisfit(self, obs, prec, idx, loc)
    
    def get_geom(self, theta: torch.Tensor, misfit_obj: DataMisfit, 
                sigma: float = 1.2, alpha: float = 0.1, s: float = 0.6) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get geometric quantities (misfit value and gradient) using automatic differentiation.
        """
        # Ensure theta requires gradients
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, device=self.device, dtype=self.dtype)
        theta = theta.requires_grad_(True)
        
        # Compute inflow profile
        inflow_profile = self.inflow_profile(theta, sigma, alpha, s)
        
        # Solve forward problem
        u, v, p, l = self.solve_forward(inflow_profile)
        
        # Compute misfit
        nll = misfit_obj.eval(u)
        
        # Compute gradient using automatic differentiation
        if theta.grad is not None:
            theta.grad.zero_()
        
        nll.backward()
        dnll = theta.grad.clone() if theta.grad is not None else None
        
        return nll, dnll
    
    def test(self, dim_theta: int = 10, sigma: float = 1.2, alpha: float = 0.1, 
             s: float = 0.6, var_obs: float = 1e-2, chk_fd: bool = False, h: float = 1e-4):
        """Test the implementation."""
        print("Testing PyTorch Laminar Jet implementation...")
        
        # Generate random theta
        theta = torch.randn(dim_theta, device=self.device, dtype=self.dtype)
        
        # Get inflow profile
        inflow_profile = self.inflow_profile(theta, sigma, alpha, s)
        print(f"Generated inflow profile with {len(inflow_profile)} points")
        
        # Get observations
        print("Obtaining observations...")
        obs, idx, loc = self.get_obs(inflow_profile)
        num_obs = len(obs)
        print(f"{num_obs} observations obtained")
        
        # Add noise
        obs += torch.sqrt(torch.tensor(var_obs, device=self.device, dtype=self.dtype)) * torch.randn_like(obs)
        
        # Reduce observations for faster computation
        obs = obs[::3]
        idx = idx[::3]
        loc = loc[::3]
        red_num_obs = len(obs)
        print(f"Reduced to {red_num_obs} observations")
        
        # Create data misfit
        misfit = self.data_misfit(obs, 1.0/var_obs, idx, loc)
        
        # Get geometric quantities
        print("\nObtaining geometric quantities with automatic differentiation...")
        import time
        start = time.time()
        
        nll, dnll = self.get_geom(theta, misfit, sigma, alpha, s)
        
        print(f"Negative log-likelihood: {nll.item():.6f}")
        if dnll is not None:
            print(f"Gradient norm: {torch.norm(dnll).item():.6f}")
            print(f"Gradient: {dnll.detach().cpu().numpy()}")
        
        end = time.time()
        print(f"Time used: {end - start:.4f} seconds")
        
        # Finite difference check
        if chk_fd and dnll is not None:
            print("\nTesting against finite difference...")
            theta_np = theta.detach().cpu().numpy()
            dnll_fd = np.zeros_like(theta_np)
            
            for i in range(len(theta_np)):
                theta_p = theta_np.copy()
                theta_m = theta_np.copy()
                theta_p[i] += h
                theta_m[i] -= h
                
                theta_p_torch = torch.tensor(theta_p, device=self.device, dtype=self.dtype)
                theta_m_torch = torch.tensor(theta_m, device=self.device, dtype=self.dtype)
                
                nll_p, _ = self.get_geom(theta_p_torch, misfit, sigma, alpha, s)
                nll_m, _ = self.get_geom(theta_m_torch, misfit, sigma, alpha, s)
                
                dnll_fd[i] = (nll_p.item() - nll_m.item()) / (2 * h)
            
            dnll_np = dnll.detach().cpu().numpy()
            diff_grad = dnll_fd - dnll_np
            print(f"Finite difference gradient: {dnll_fd}")
            print(f"Gradient difference (inf-norm): {np.linalg.norm(diff_grad, np.inf):.10f}")
            print(f"Gradient difference (2-norm): {np.linalg.norm(diff_grad):.10f}")


if __name__ == '__main__':
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    laminar = LaminarPyTorch(unit=0.1, nx=20, ny=20, nu=4.0e-2, stokes=True, device=device)
    
    # Run test
    laminar.test(dim_theta=10, sigma=0.1, alpha=1, chk_fd=True)