import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class LaminarJetSimulator:
    def __init__(self, nx=128, ny=128, Lx=10.0, Ly=8.0, nu=1e-2, dt=1e-3, nt=1000, device='cpu'):
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx, self.dy = Lx / nx, Ly / ny
        self.nu = nu
        self.dt = dt
        self.nt = nt
        self.device = device

        # grid
        self.x = torch.linspace(0, Lx, nx, device=device)
        self.y = torch.linspace(0, Ly, ny, device=device)
        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing='ij')

        # velocity and pressure fields
        self.u = torch.zeros((nx, ny), device=device, requires_grad=True)
        self.v = torch.zeros((nx, ny), device=device, requires_grad=True)
        self.p = torch.zeros((nx, ny), device=device, requires_grad=True)

    def apply_boundary_conditions(self):
        y0, r = 4.0, 0.5
        inflow = torch.where(
            (self.y >= y0 - r) & (self.y <= y0 + r),
            1.0 - ((self.y - y0) / r) ** 2,
            torch.zeros_like(self.y)
        )
        self.u[0, :] = inflow
        self.v[0, :] = 0.0
        self.u[:, 0] = self.u[:, -1] = 0.0
        self.v[:, 0] = self.v[:, -1] = 0.0
        self.u[-1, :] = self.u[-2, :]
        self.v[-1, :] = self.v[-2, :]

    def pressure_projection(self, tol=1e-6, max_iter=10000):
        div = (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2 * self.dx) + (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2 * self.dy)
        rhs = div / self.dt

        p = torch.zeros_like(self.p)
        for _ in range(max_iter):
            p_new = p.clone()
            p_new[1:-1, 1:-1] = 0.25 * (
                p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - self.dx * self.dy * rhs
            )
            if (p_new - p).abs().max() < tol:
                break
            p = p_new

        # Apply Neumann BCs on all walls
        p[0, :] = p[1, :]     # left
        p[-1, :] = p[-2, :]   # right (outlet)
        p[:, 0] = p[:, 1]     # bottom
        p[:, -1] = p[:, -2]   # top

        self.p = p

        # Subtract pressure gradient
        self.u[1:-1, 1:-1] -= self.dt * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * self.dx)
        self.v[1:-1, 1:-1] -= self.dt * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * self.dy)

    def replicate_pad_2d(self, field):
        left = field[:, :1]
        right = field[:, -1:]
        field = torch.cat([left, field, right], dim=1)
        top = field[:1, :]
        bottom = field[-1:, :]
        field = torch.cat([top, field, bottom], dim=0)
        return field

    def fixed_point_step(self):
        u, v, dx, dy, dt, nu = self.u, self.v, self.dx, self.dy, self.dt, self.nu

        u_pad = self.replicate_pad_2d(u[1:-1, 1:-1])
        v_pad = self.replicate_pad_2d(v[1:-1, 1:-1])

        u_xx = (u_pad[2:, 1:-1] - 2 * u_pad[1:-1, 1:-1] + u_pad[:-2, 1:-1]) / dx**2
        u_yy = (u_pad[1:-1, 2:] - 2 * u_pad[1:-1, 1:-1] + u_pad[1:-1, :-2]) / dy**2

        v_xx = (v_pad[2:, 1:-1] - 2 * v_pad[1:-1, 1:-1] + v_pad[:-2, 1:-1]) / dx**2
        v_yy = (v_pad[1:-1, 2:] - 2 * v_pad[1:-1, 1:-1] + v_pad[1:-1, :-2]) / dy**2

        u_center = u[1:-1, 1:-1]
        v_center = v[1:-1, 1:-1]

        u_adv = u_center * (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx) + v_center * (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dy)
        v_adv = u_center * (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dx) + v_center * (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dy)

        u_new = u_center + dt * (-u_adv + nu * (u_xx + u_yy))
        v_new = v_center + dt * (-v_adv + nu * (v_xx + v_yy))

        return u_new, v_new

    def step(self):
        self.u = self.u.clone()
        self.v = self.v.clone()
        self.u[1:-1, 1:-1], self.v[1:-1, 1:-1] = self.fixed_point_step()
        self.pressure_projection()
        self.apply_boundary_conditions()

    def run(self):
        residuals = []
        for _ in range(self.nt):
            u_old, v_old = self.u.clone(), self.v.clone()
            self.step()
            res = ((self.u - u_old)**2 + (self.v - v_old)**2).mean().item()
            residuals.append(res)

        # Plot residuals
        plt.semilogy(residuals)
        plt.title("Velocity Residuals")
        plt.xlabel("Step")
        plt.ylabel("Residual Norm")
        plt.grid(True)
        plt.savefig("velocity_residuals.png")
        plt.close()

        return self.u, self.v

if __name__ == '__main__':
    sim = LaminarJetSimulator(device='cpu')
    u, v = sim.run()

    # 1. Check incompressibility
    with torch.no_grad():
        div = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * sim.dx) + (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * sim.dy)
        print("Max divergence:", div.abs().max().item())

    # 2. Visualize flow field
    plt.figure()
    plt.quiver(sim.X[::4, ::4].detach().cpu(), sim.Y[::4, ::4].detach().cpu(), u[::4, ::4].detach().cpu(), v[::4, ::4].detach().cpu())
    plt.title("Velocity Field")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.savefig("velocity_field_quiver.png")
    plt.close()

    # 3. Visualize pressure + velocity overlay
    plt.figure(figsize=(10, 6))
    plt.imshow(sim.p.T.detach().cpu(), origin="lower", extent=[0, sim.Lx, 0, sim.Ly], cmap="jet", aspect="auto", interpolation='bilinear')
    plt.colorbar(label="Pressure")
    plt.quiver(sim.X[::4, ::4].detach().cpu(), sim.Y[::4, ::4].detach().cpu(), u[::4, ::4].detach().cpu(), v[::4, ::4].detach().cpu(), color="black")
    plt.title("Pressure and Velocity Field (Laminar Jet)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("pressure_velocity_figure11.png")
    plt.close()

    # 4. Final max velocity printout
    print("Final u max:", u.max().item())
    print("Final v max:", v.max().item())
