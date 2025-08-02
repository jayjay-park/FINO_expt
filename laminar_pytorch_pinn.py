#!/usr/bin/env python
"""
PyTorch implementation of Laminar-Jet PDE model using Physics-Informed Neural Networks (PINNs)
Converted from FEniCS implementation to be fully differentiable
-----------------------------------
This implementation uses PINNs to solve the Navier-Stokes equations directly,
making the entire pipeline differentiable through automatic differentiation.
-----------------------------------
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import time

class LaminarPINN(nn.Module):
    """
    Physics-Informed Neural Network implementation of the Laminar Jet PDE model.
    """
    
    def __init__(self, 
                 unit: float = 1.0,
                 nx: int = 40, 
                 ny: int = 30,
                 nu: float = 1.0e-1,
                 beta: float = 0.5,
                 stokes: bool = False,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32,
                 hidden_layers: List[int] = [128, 128, 128, 128]):
        super(LaminarPINN, self).__init__()
        
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
        
        # Device and dtype
        self.device = device
        self.dtype = dtype
        
        # Setup domain and training points
        self.setup_domain()
        
        # Neural network architecture
        layers = [2] + hidden_layers + [3]  # Input: (x,y), Output: (u,v,p)
        self.layers = nn.ModuleList()
        
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            
        # Activation function
        self.activation = nn.Tanh()
        
        # Initialize weights
        self.init_weights()
        
        # Training parameters
        self.pde_weight = 1.0
        self.bc_weight = 100.0
        self.data_weight = 1000.0
        
    def init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def setup_domain(self):
        """Setup the computational domain and training points."""
        # Domain boundaries
        x_min, x_max = 0.0, self.Lx
        y_min, y_max = -0.5 * self.Ly, 0.5 * self.Ly
        
        # Interior points for PDE residual
        n_interior = self.nx * self.ny
        x_interior = torch.rand(n_interior, device=self.device, dtype=self.dtype) * (x_max - x_min) + x_min
        y_interior = torch.rand(n_interior, device=self.device, dtype=self.dtype) * (y_max - y_min) + y_min
        self.interior_points = torch.stack([x_interior, y_interior], dim=1)
        
        # Boundary points
        n_bc = 100
        
        # Inlet boundary (x = 0)
        y_inlet = torch.linspace(y_min, y_max, n_bc, device=self.device, dtype=self.dtype)
        x_inlet = torch.zeros_like(y_inlet)
        self.inlet_points = torch.stack([x_inlet, y_inlet], dim=1)
        self.inlet_y_coords = y_inlet
        
        # Outlet boundary (x = Lx)
        y_outlet = torch.linspace(y_min, y_max, n_bc, device=self.device, dtype=self.dtype)
        x_outlet = torch.full_like(y_outlet, x_max)
        self.outlet_points = torch.stack([x_outlet, y_outlet], dim=1)
        
        # Top and bottom boundaries
        x_wall = torch.linspace(x_min, x_max, n_bc, device=self.device, dtype=self.dtype)
        y_top = torch.full_like(x_wall, y_max)
        y_bottom = torch.full_like(x_wall, y_min)
        self.top_points = torch.stack([x_wall, y_top], dim=1)
        self.bottom_points = torch.stack([x_wall, y_bottom], dim=1)
        
        # Normalization factors for inputs
        self.x_norm = x_max - x_min
        self.y_norm = y_max - y_min
        self.x_offset = x_min
        self.y_offset = y_min
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input coordinates to [-1, 1]."""
        x_norm = x.clone()
        x_norm[:, 0] = 2 * (x_norm[:, 0] - self.x_offset) / self.x_norm - 1
        x_norm[:, 1] = 2 * (x_norm[:, 1] - self.y_offset) / self.y_norm - 1
        return x_norm
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network."""
        x_norm = self.normalize_input(x)
        
        for i, layer in enumerate(self.layers[:-1]):
            x_norm = self.activation(layer(x_norm))
        
        # Final layer without activation
        output = self.layers[-1](x_norm)
        
        return output
    
    def get_solution(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get velocity and pressure fields."""
        output = self.forward(x)
        u = output[:, 0:1]
        v = output[:, 1:2]
        p = output[:, 2:3]
        return u.squeeze(), v.squeeze(), p.squeeze()
    
    def inflow_profile(self, theta: torch.Tensor, sigma: float = 1.2, 
                      alpha: float = 0.1, s: float = 0.6) -> torch.Tensor:
        """
        Karhunen-Loeve expansion of inflow profile.
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
    
    def compute_pde_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Compute PDE residual for interior points."""
        x.requires_grad_(True)
        
        u, v, p = self.get_solution(x)
        
        # Compute gradients
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 0]
        u_y = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 1]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, 0]
        v_y = torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, 1]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0][:, 0]
        p_y = torch.autograd.grad(p.sum(), x, create_graph=True)[0][:, 1]
        
        # Compute second derivatives for Laplacian
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0][:, 1]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0][:, 0]
        v_yy = torch.autograd.grad(v_y.sum(), x, create_graph=True)[0][:, 1]
        
        # PDE residuals
        if self.stokes:
            # Linear Stokes equations
            momentum_x = self.nu * (u_xx + u_yy) - p_x
            momentum_y = self.nu * (v_xx + v_yy) - p_y
        else:
            # Nonlinear Navier-Stokes equations
            momentum_x = self.nu * (u_xx + u_yy) - u * u_x - v * u_y - p_x
            momentum_y = self.nu * (v_xx + v_yy) - u * v_x - v * v_y - p_y
        
        # Continuity equation
        continuity = u_x + v_y
        
        return momentum_x, momentum_y, continuity
    
    def compute_boundary_loss(self, inflow_profile: torch.Tensor) -> torch.Tensor:
        """Compute boundary condition losses."""
        total_bc_loss = 0.0
        
        # Inlet boundary conditions
        u_inlet, v_inlet, _ = self.get_solution(self.inlet_points)
        inlet_u_loss = torch.mean((u_inlet - inflow_profile)**2)
        inlet_v_loss = torch.mean(v_inlet**2)
        total_bc_loss += inlet_u_loss + inlet_v_loss
        
        # Wall boundary conditions (no-slip: u = v = 0)
        u_top, v_top, _ = self.get_solution(self.top_points)
        u_bottom, v_bottom, _ = self.get_solution(self.bottom_points)
        
        wall_loss = torch.mean(u_top**2) + torch.mean(v_top**2) + \
                   torch.mean(u_bottom**2) + torch.mean(v_bottom**2)
        total_bc_loss += wall_loss
        
        return total_bc_loss
    
    def compute_pde_loss(self) -> torch.Tensor:
        """Compute PDE residual loss."""
        momentum_x, momentum_y, continuity = self.compute_pde_residual(self.interior_points)
        
        pde_loss = torch.mean(momentum_x**2) + torch.mean(momentum_y**2) + torch.mean(continuity**2)
        
        return pde_loss
    
    def train_pinn(self, inflow_profile: torch.Tensor, 
                   epochs: int = 5000, lr: float = 1e-3, 
                   print_every: int = 500) -> List[float]:
        """Train the PINN to solve the PDE."""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        
        losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute losses
            pde_loss = self.compute_pde_loss()
            bc_loss = self.compute_boundary_loss(inflow_profile)
            
            # Total loss
            total_loss = self.pde_weight * pde_loss + self.bc_weight * bc_loss
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            losses.append(total_loss.item())
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}, "
                      f"PDE Loss = {pde_loss.item():.6f}, BC Loss = {bc_loss.item():.6f}")
        
        return losses
    
    def solve_forward(self, inflow_profile: torch.Tensor, 
                     train_epochs: int = 2000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Solve the forward PDE problem.
        """
        # Train the PINN
        losses = self.train_pinn(inflow_profile, epochs=train_epochs, print_every=500)
        
        # Get solution at all domain points
        domain_points = torch.cat([
            self.interior_points,
            self.inlet_points,
            self.outlet_points,
            self.top_points,
            self.bottom_points
        ], dim=0)
        
        with torch.no_grad():
            u, v, p = self.get_solution(domain_points)
        
        return u, v, p
    
    def get_obs(self, inflow_profile: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get observations at outlet boundary."""
        # Solve forward problem
        u, v, p = self.solve_forward(inflow_profile)
        
        # Get solution at outlet points
        with torch.no_grad():
            u_outlet, _, _ = self.get_solution(self.outlet_points)
        
        # Create indices for outlet points
        outlet_indices = torch.arange(len(self.outlet_points), device=self.device)
        
        return u_outlet, outlet_indices, self.outlet_points
    
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
            if self.idx is not None and len(self.idx) < len(u):
                u_obs = u[self.idx]
            else:
                u_obs = u
            
            diff = u_obs - self.obs
            misfit = 0.5 * self.prec * torch.sum(diff**2)
            return misfit
    
    def data_misfit(self, obs: torch.Tensor, prec: float, 
                   idx: Optional[torch.Tensor] = None, 
                   loc: Optional[torch.Tensor] = None) -> DataMisfit:
        """Create data misfit object."""
        return self.DataMisfit(self, obs, prec, idx, loc)
    
    def get_geom(self, theta: torch.Tensor, misfit_obj: DataMisfit, 
                sigma: float = 1.2, alpha: float = 0.1, s: float = 0.6,
                train_epochs: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get geometric quantities (misfit value and gradient) using automatic differentiation.
        """
        # Ensure theta requires gradients
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, device=self.device, dtype=self.dtype)
        theta = theta.requires_grad_(True)
        
        # Compute inflow profile
        inflow_profile = self.inflow_profile(theta, sigma, alpha, s)
        
        # Train PINN for this specific inflow profile
        self.train_pinn(inflow_profile, epochs=train_epochs, print_every=200)
        
        # Get solution at outlet points
        u_outlet, _, _ = self.get_solution(self.outlet_points)
        
        # Compute misfit
        nll = misfit_obj.eval(u_outlet)
        
        # Compute gradient using automatic differentiation
        if theta.grad is not None:
            theta.grad.zero_()
        
        nll.backward()
        dnll = theta.grad.clone() if theta.grad is not None else None
        
        return nll, dnll
    
    def plot_solution(self, inflow_profile: torch.Tensor, save_path: Optional[str] = None):
        """Plot the solution fields."""
        # Create a finer grid for plotting
        nx_plot, ny_plot = 50, 40
        x_plot = torch.linspace(0, self.Lx, nx_plot, device=self.device, dtype=self.dtype)
        y_plot = torch.linspace(-0.5 * self.Ly, 0.5 * self.Ly, ny_plot, device=self.device, dtype=self.dtype)
        X_plot, Y_plot = torch.meshgrid(x_plot, y_plot, indexing='ij')
        plot_points = torch.stack([X_plot.flatten(), Y_plot.flatten()], dim=1)
        
        # Train PINN and get solution
        self.train_pinn(inflow_profile, epochs=1000, print_every=200)
        
        with torch.no_grad():
            u, v, p = self.get_solution(plot_points)
        
        # Convert to numpy for plotting
        X_np = X_plot.cpu().numpy()
        Y_np = Y_plot.cpu().numpy()
        u_np = u.cpu().numpy().reshape(nx_plot, ny_plot)
        v_np = v.cpu().numpy().reshape(nx_plot, ny_plot)
        p_np = p.cpu().numpy().reshape(nx_plot, ny_plot)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Velocity magnitude
        vel_mag = np.sqrt(u_np**2 + v_np**2)
        im1 = axes[0, 0].contourf(X_np, Y_np, vel_mag, levels=20, cmap='viridis')
        axes[0, 0].set_title('Velocity Magnitude')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # u-velocity
        im2 = axes[0, 1].contourf(X_np, Y_np, u_np, levels=20, cmap='RdBu_r')
        axes[0, 1].set_title('u-velocity')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # v-velocity
        im3 = axes[1, 0].contourf(X_np, Y_np, v_np, levels=20, cmap='RdBu_r')
        axes[1, 0].set_title('v-velocity')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Pressure
        im4 = axes[1, 1].contourf(X_np, Y_np, p_np, levels=20, cmap='coolwarm')
        axes[1, 1].set_title('Pressure')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def test(self, dim_theta: int = 10, sigma: float = 1.2, alpha: float = 0.1, 
             s: float = 0.6, var_obs: float = 1e-2, chk_fd: bool = False, 
             h: float = 1e-4, plot_solution: bool = True):
        """Test the implementation."""
        print("Testing PyTorch PINN Laminar Jet implementation...")
        
        # Generate random theta
        theta = torch.randn(dim_theta, device=self.device, dtype=self.dtype)
        print(f"Generated theta: {theta.cpu().numpy()}")
        
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
        idx = idx[::3] if idx is not None else None
        loc = loc[::3] if loc is not None else None
        red_num_obs = len(obs)
        print(f"Reduced to {red_num_obs} observations")
        
        # Create data misfit
        misfit = self.data_misfit(obs, 1.0/var_obs, idx, loc)
        
        # Get geometric quantities
        print("\nObtaining geometric quantities with automatic differentiation...")
        start = time.time()
        
        nll, dnll = self.get_geom(theta, misfit, sigma, alpha, s, train_epochs=500)
        
        print(f"Negative log-likelihood: {nll.item():.6f}")
        if dnll is not None:
            print(f"Gradient norm: {torch.norm(dnll).item():.6f}")
            print(f"Gradient: {dnll.detach().cpu().numpy()}")
        
        end = time.time()
        print(f"Time used: {end - start:.4f} seconds")
        
        # Plot solution
        if plot_solution:
            print("\nPlotting solution...")
            self.plot_solution(inflow_profile)
        
        # Finite difference check (simplified for PINN)
        if chk_fd and dnll is not None:
            print("\nTesting against finite difference (simplified)...")
            theta_np = theta.detach().cpu().numpy()
            dnll_fd = np.zeros_like(theta_np)
            
            # Test only first few components for speed
            n_test = min(3, len(theta_np))
            for i in range(n_test):
                theta_p = theta_np.copy()
                theta_m = theta_np.copy()
                theta_p[i] += h
                theta_m[i] -= h
                
                theta_p_torch = torch.tensor(theta_p, device=self.device, dtype=self.dtype)
                theta_m_torch = torch.tensor(theta_m, device=self.device, dtype=self.dtype)
                
                # Use fewer training epochs for FD check
                nll_p, _ = self.get_geom(theta_p_torch, misfit, sigma, alpha, s, train_epochs=200)
                nll_m, _ = self.get_geom(theta_m_torch, misfit, sigma, alpha, s, train_epochs=200)
                
                dnll_fd[i] = (nll_p.item() - nll_m.item()) / (2 * h)
            
            dnll_np = dnll.detach().cpu().numpy()
            diff_grad = dnll_fd[:n_test] - dnll_np[:n_test]
            print(f"Finite difference gradient (first {n_test} components): {dnll_fd[:n_test]}")
            print(f"AD gradient (first {n_test} components): {dnll_np[:n_test]}")
            print(f"Gradient difference (inf-norm): {np.linalg.norm(diff_grad, np.inf):.6f}")
            print(f"Gradient difference (2-norm): {np.linalg.norm(diff_grad):.6f}")


if __name__ == '__main__':
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    laminar = LaminarPINN(unit=0.1, nx=30, ny=25, nu=4.0e-2, stokes=True, device=device)
    
    # Run test
    laminar.test(dim_theta=5, sigma=0.1, alpha=1, chk_fd=False, plot_solution=True)