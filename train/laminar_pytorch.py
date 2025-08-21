import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class LaminarJetSimulator:
    def __init__(self, nx=128, ny=128, Lx=1.0, Ly=0.8, nu=1e-2, dt=1e-3, nt=1000,
                 coeffs=None, device='cpu'):
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx, self.dy = Lx / nx, Ly / ny
        self.nu = nu
        self.dt = dt
        self.nt = nt
        self.device = device

        # inflow coefficients (parameter vector)
        if coeffs is None:
            # default 3-mode profile
            coeffs = torch.tensor([1.0, -0.5, 0.25], device=device)
        self.coeffs = coeffs.to(device)

        # grid
        self.x = torch.linspace(0, Lx, nx, device=device)
        self.y = torch.linspace(-Ly/2, Ly/2, ny, device=device)  # domain: [-Ly/2, Ly/2]
        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing='ij')

        # velocity and pressure fields
        self.u = torch.zeros((nx, ny), device=device, requires_grad=True)
        self.v = torch.zeros((nx, ny), device=device, requires_grad=True)
        self.p = torch.zeros((nx, ny), device=device, requires_grad=True)

    def inflow_profile(self, coeffs):
        """
        Construct inflow profile θ(y) from cosine basis.
        coeffs: (r,) tensor
        returns: (ny,) inflow velocity profile
        """
        r = len(coeffs)
        basis = torch.stack([
            torch.cos((k+1) * torch.pi * self.y / self.Ly)
            for k in range(r)
        ], dim=0)  # (r, ny)
        return coeffs @ basis   # shape (ny,)

    def apply_boundary_conditions(self):
        inflow = self.inflow_profile(self.coeffs)
        self.u[0, :] = inflow
        self.v[0, :] = 0.0
        # free-slip top/bottom
        self.u[:, 0] = self.u[:, -1] = 0.0
        self.v[:, 0] = self.v[:, -1] = 0.0
        # outflow copy
        self.u[-1, :] = self.u[-2, :]
        self.v[-1, :] = self.v[-2, :]

    def pressure_projection(self):
        div = (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2 * self.dx) + \
              (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2 * self.dy)
        rhs = div / self.dt

        p = torch.zeros_like(self.p)
        for _ in range(200):
            p_new = p.clone()
            p_new[1:-1, 1:-1] = 0.25 * (
                p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] -
                self.dx * self.dy * rhs
            )
            p = p_new
        p[-1, :] = p[-2, :]
        self.p = p

        self.u[1:-1, 1:-1] -= self.dt * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * self.dx)
        self.v[1:-1, 1:-1] -= self.dt * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * self.dy)

    def replicate_pad_2d(self, field):
        left = field[:, :1]; right = field[:, -1:]
        field = torch.cat([left, field, right], dim=1)
        top = field[:1, :]; bottom = field[-1:, :]
        field = torch.cat([top, field, bottom], dim=0)
        return field

    def step(self):
        u, v, dx, dy, dt, nu = self.u, self.v, self.dx, self.dy, self.dt, self.nu
        u_pad = self.replicate_pad_2d(u[1:-1, 1:-1])
        v_pad = self.replicate_pad_2d(v[1:-1, 1:-1])

        u_xx = (u_pad[2:, 1:-1] - 2*u_pad[1:-1, 1:-1] + u_pad[:-2, 1:-1]) / dx**2
        u_yy = (u_pad[1:-1, 2:] - 2*u_pad[1:-1, 1:-1] + u_pad[1:-1, :-2]) / dy**2
        v_xx = (v_pad[2:, 1:-1] - 2*v_pad[1:-1, 1:-1] + v_pad[:-2, 1:-1]) / dx**2
        v_yy = (v_pad[1:-1, 2:] - 2*v_pad[1:-1, 1:-1] + v_pad[1:-1, :-2]) / dy**2

        u_center = u[1:-1, 1:-1]; v_center = v[1:-1, 1:-1]
        u_adv = u_center * (u[2:, 1:-1] - u[:-2, 1:-1])/(2*dx) + \
                v_center * (u[1:-1, 2:] - u[1:-1, :-2])/(2*dy)
        v_adv = u_center * (v[2:, 1:-1] - v[:-2, 1:-1])/(2*dx) + \
                v_center * (v[1:-1, 2:] - v[1:-1, :-2])/(2*dy)

        u_new = u_center + dt * (-u_adv + nu*(u_xx+u_yy))
        v_new = v_center + dt * (-v_adv + nu*(v_xx+v_yy))

        self.u = u.clone(); self.v = v.clone()
        self.u[1:-1, 1:-1] = u_new; self.v[1:-1, 1:-1] = v_new

        self.pressure_projection()
        self.apply_boundary_conditions()

    def run(self):
        for _ in range(self.nt):
            self.step()
        return self.u, self.v
        
    def plot(self, save_path=None, stride=3):
        """
        Match Fig. 11 (top right) from Beskos et al.:
        velocity magnitude heatmap + velocity quiver.
        """
        u = self.u.detach().cpu()
        v = self.v.detach().cpu()
        vel_mag = torch.sqrt(u**2 + v**2)

        X = self.X.cpu()
        Y = self.Y.cpu()

        plt.figure(figsize=(6, 4))
        pcm = plt.contourf(X, Y, vel_mag, levels=50, cmap="jet",
                           vmin=-0.3, vmax=0.18)
        plt.colorbar(pcm, label="‖u‖")

        plt.quiver(X[::stride, ::stride], Y[::stride, ::stride],
                   u[::stride, ::stride], v[::stride, ::stride],
                   color="k", scale=30)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Solutions of velocity and pressure (Fig. 11 style)")
        plt.gca().set_aspect("equal")

        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()




    def observe(self, n_sensors=7):
        """Return outlet velocities at n_sensors points"""
        ys = torch.linspace(-self.Ly/2, self.Ly/2, n_sensors, device=self.device)
        indices = ((ys + self.Ly/2)/self.Ly * (self.ny-1)).long()
        return self.u[-1, indices]


if __name__ == "__main__":
    coeffs = torch.tensor([1.0, -0.5, 0.2])  # inflow coefficients
    sim = LaminarJetSimulator(coeffs=coeffs, device="cpu")

    u, v = sim.run()
    obs = sim.observe()
    print("Outlet observations:", obs)

    sim.plot("laminar_jet.png")
