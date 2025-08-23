import torch
import math
import matplotlib.pyplot as plt
from PINO_NS import NavierStokes2d, GaussianRF   # import the real solver

class NavierStokesSimulator(torch.nn.Module):
    """
    Wrapper around PINO_NS.NavierStokes2d so it can be used interchangeably.
    Provides same API as before but delegates to inner_solver.
    """

    def __init__(self, s1, s2, Re=100, T=1, delta_t=1e-3, adaptive=False):
        super().__init__()
        self.s1, self.s2 = s1, s2
        self.Re = Re
        self.adaptive = adaptive
        self.delta_t = delta_t
        self.T = T,
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float64

        # instantiate the original PINO_NS solver
        self.inner_solver = NavierStokes2d(s1, s2,
                                           L1=2*math.pi,
                                           L2=2*math.pi,
                                           device=self.device)

        # store forcing
        t = torch.linspace(0, 1, s1 + 1, device=self.device)[:-1]
        X, Y = torch.meshgrid(t, t, indexing="ij")
        self.f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) +
                        torch.cos(2 * math.pi * (X + Y)))

    def forward(self, w, T=1.0, Re=None, adaptive=None, delta_t=None):
        """
        Just forward to PINO_NS.NavierStokes2d
        """
        if isinstance(self.T, (tuple, list)):
            T = self.T[0]
        return self.inner_solver(w,
                                 f=self.f,
                                 T=T,
                                 Re=self.Re or Re)

    __call__ = forward  # make it callable

    def sample(self, N=1, alpha=2.5, tau=7.0, sigma=None, mean=None):
        """
        Use GaussianRF2d from PINO_NS.
        """
        grf = GaussianRF(2, self.s1,
                           alpha=alpha, tau=tau, device=self.device)
        u = grf.sample(N)
        return u if N > 1 else u[0]

    def plot_vorticity(self, w, step=0, file_path="vorticity.png", title="Vorticity Field"):
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(w, cmap="jet", origin="lower", extent=[0, 1., 0, 1.])
        fig.colorbar(im, ax=ax, label="Vorticity", fraction=0.045, pad=0.06)
        ax.set_title(title)
        fig.savefig(file_path.replace(".png", f"_{step}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    
    def plot_data(self, x, y, v, Jvp, file_path="plot.png", title="NS Sample Plot"):
        """
        Plot velocity fields and their curls for visualization.
        
        Args:
            x: Input velocity fields
            y: Output velocity fields
            v: Eigenvector velocity fields
            Jvp: Jacobian-vector product velocity fields
            file_path: Path to save the plot
        """
        def prepare_field(field):
            field = field.reshape(x.shape)
            return {
                'vorticity': field.cpu().numpy()
            }
        
        # Prepare all fields
        x_data = prepare_field(x)
        y_data = prepare_field(y)
        v_data = prepare_field(v)
        jvp_data = prepare_field(Jvp)
        
        # Create figure and subplots
        fig, axs = plt.subplots(4, 1, figsize=(5, 18))
        fig.suptitle(title)
        
        # Data to plot with corresponding titles
        plot_data = [
            (0, x_data, ['Input']),
            (1, y_data, ['Output']),
            (2, v_data, ['Eigenvector']),
            (3, jvp_data, ['Jvp'])
        ]
        
        # Plot all data
        for row, data, titles in plot_data:
            if row > 1:
                cmap = 'BuPu'
            else:
                cmap = 'jet' #'viridis'
            im = axs[row].imshow(data['vorticity'], cmap=cmap)
            if row == 1:
                # choose 10 levels between min and max of the field
                levels = np.linspace(data['vorticity'].min(), data['vorticity'].max(), 10)
                axs[row].contour(data['vorticity'], levels=levels, colors='coral', linewidths=2.0)
            fig.colorbar(im, ax=axs[row], fraction=0.046) 
            axs[row].set_title(titles[0])
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    @property
    def domain(self):
        return self.s1 * self.s2

    @property
    def range(self):
        return self.s1 * self.s2


# import torch
# import torch.fft as fft
# import numpy as np
# import math
# import matplotlib.pyplot as plt


# class GaussianRF(object):
#     def __init__(self, dim, size, length=1.0, alpha=2.0, tau=3.0, sigma=None, boundary="periodic", constant_eig=False, device=None):

#         self.dim = dim
#         self.device = device

#         if sigma is None:
#             sigma = tau**(0.5*(2*alpha - self.dim))

#         k_max = size//2

#         const = (4*(math.pi**2))/(length**2)

#         if dim == 1:
#             k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
#                            torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

#             self.sqrt_eig = size*math.sqrt(2.0)*sigma*((const*(k**2) + tau**2)**(-alpha/2.0))

#             if constant_eig:
#                 self.sqrt_eig[0] = size*sigma*(tau**(-alpha))
#             else:
#                 self.sqrt_eig[0] = 0.0

#         elif dim == 2:
#             wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
#                                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

#             k_x = wavenumers.transpose(0,1)
#             k_y = wavenumers

#             self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((const*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))

#             if constant_eig:
#                 self.sqrt_eig[0,0] = (size**2)*sigma*(tau**(-alpha))
#             else:
#                 self.sqrt_eig[0,0] = 0.0

#         elif dim == 3:
#             wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
#                                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

#             k_x = wavenumers.transpose(1,2)
#             k_y = wavenumers
#             k_z = wavenumers.transpose(0,2)

#             self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((const*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))

#             if constant_eig:
#                 self.sqrt_eig[0,0,0] = (size**3)*sigma*(tau**(-alpha))
#             else:
#                 self.sqrt_eig[0,0,0] = 0.0

#         self.size = []
#         for j in range(self.dim):
#             self.size.append(size)

#         self.size = tuple(self.size)

#     def sample(self, N):

#         coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
#         coeff = self.sqrt_eig*coeff

#         u = torch.fft.irfftn(coeff, self.size, norm="backward")
#         return u

# class NavierStokesSimulator(torch.nn.Module):
#     """
#     Navier–Stokes simulator using PINO_NS.py scheme,
#     wrapped in NS.py-style class with helpers.
#     """

#     def __init__(self,
#                  s1,
#                  s2,
#                  scale=10.0,
#                  Re=100,
#                  adaptive=False,
#                  delta_t=1e-3,
#                  nburn=10):

#         super().__init__()
#         self.s1 = s1
#         self.s2 = s2
#         self.scale = scale
#         self.Re = Re
#         self.adaptive = adaptive
#         self.delta_t = delta_t
#         self.nburn = nburn

#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.dtype = torch.float64

#         # Domain
#         self.L1 = 2 * math.pi
#         self.L2 = 2 * math.pi
#         self.h = 1.0 / max(s1, s2)

#         # Forcing
#         t = torch.linspace(0, 1, s1 + 1, device=self.device)
#         t = t[:-1]
#         X, Y = torch.meshgrid(t, t, indexing="ij")
#         self.f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) +
#                         torch.cos(2 * math.pi * (X + Y)))

#         # Spectral operators
#         freq_list1 = torch.cat((torch.arange(0, s1 // 2, step=1),
#                                 torch.zeros((1,)),
#                                 torch.arange(-s1 // 2 + 1, 0, step=1)), 0)
#         self.k1 = freq_list1.view(-1, 1).repeat(1, s2 // 2 + 1).type(self.dtype).to(self.device)

#         freq_list2 = torch.cat((torch.arange(0, s2 // 2, step=1),
#                                 torch.zeros((1,))), 0)
#         self.k2 = freq_list2.view(1, -1).repeat(s1, 1).type(self.dtype).to(self.device)

#         # Laplacian
#         freq_list1 = torch.cat((torch.arange(0, s1 // 2, step=1),
#                                 torch.arange(-s1 // 2, 0, step=1)), 0)
#         k1 = freq_list1.view(-1, 1).repeat(1, s2 // 2 + 1).type(self.dtype).to(self.device)

#         freq_list2 = torch.arange(0, s2 // 2 + 1, step=1)
#         k2 = freq_list2.view(1, -1).repeat(s1, 1).type(self.dtype).to(self.device)

#         self.G = ((4 * math.pi ** 2) / (self.L1 ** 2)) * k1 ** 2 + ((4 * math.pi ** 2) / (self.L2 ** 2)) * k2 ** 2

#         self.inv_lap = self.G.clone()
#         self.inv_lap[0, 0] = 1.0
#         self.inv_lap = 1.0 / self.inv_lap

#         # Dealiasing mask
#         self.dealias = (self.k1 ** 2 + self.k2 ** 2 <= 0.6 * (0.25 * s1 ** 2 + 0.25 * s2 ** 2)).type(self.dtype).to(self.device)
#         self.dealias[0, 0] = 0.0

#     # === PINO_NS core ===
#     def stream_function(self, w_h, real_space=False):
#         psi_h = self.inv_lap * w_h
#         return fft.irfft2(psi_h, s=(self.s1, self.s2)) if real_space else psi_h

#     def velocity_field(self, stream_f, real_space=True):
#         q_h = (2 * math.pi / self.L2) * 1j * self.k2 * stream_f
#         v_h = -(2 * math.pi / self.L1) * 1j * self.k1 * stream_f
#         if real_space:
#             return fft.irfft2(q_h, s=(self.s1, self.s2)), fft.irfft2(v_h, s=(self.s1, self.s2))
#         else:
#             return q_h, v_h

#     def nonlinear_term(self, w_h, f_h=None):
#         w = fft.irfft2(w_h, s=(self.s1, self.s2))
#         q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
#         nonlin = -1j * ((2 * math.pi / self.L1) * self.k1 * fft.rfft2(q * w) +
#                         (2 * math.pi / self.L2) * self.k2 * fft.rfft2(v * w))
#         if f_h is not None:
#             nonlin += f_h
#         return nonlin

#     def time_step(self, q, v, f):
#         max_speed = torch.max(torch.sqrt(q ** 2 + v ** 2)).item()
#         xi = torch.sqrt(torch.max(torch.abs(f))).item() if f is not None else 1.0
#         mu = (1.0 / self.Re) * xi * ((self.L1 / (2 * math.pi)) ** 0.75) * ((self.L2 / (2 * math.pi)) ** 0.75)
#         if max_speed == 0:
#             return 0.5 * (self.h ** 2) / mu
#         return min(0.5 * self.h / max_speed, 0.5 * (self.h ** 2) / mu)

#     # === Public API ===
#     def forward(self, w, T=1.0):
#         """Advance solution by exactly T units of time (like PINO_NS)."""
#         GG = (1.0 / self.Re) * self.G
#         w_h = fft.rfft2(w)
#         f_h = fft.rfft2(self.f) if self.f is not None else None
#         delta_t = self.delta_t

#         if self.adaptive:
#             q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
#             delta_t = self.time_step(q, v, self.f)

#         time = 0.0
#         while time < T:
#             if time + delta_t > T:
#                 current_delta_t = T - time
#             else:
#                 current_delta_t = delta_t

#             nonlin1 = self.nonlinear_term(w_h, f_h)
#             w_h_tilde = (w_h + current_delta_t * (nonlin1 - 0.5 * GG * w_h)) / (1.0 + 0.5 * current_delta_t * GG)

#             nonlin2 = self.nonlinear_term(w_h_tilde, f_h)
#             w_h = (w_h + current_delta_t * (0.5 * (nonlin1 + nonlin2) - 0.5 * GG * w_h)) / (1.0 + 0.5 * current_delta_t * GG)

#             w_h *= self.dealias
#             time += current_delta_t

#             if self.adaptive:
#                 q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
#                 delta_t = self.time_step(q, v, self.f)

#         return fft.irfft2(w_h, s=(self.s1, self.s2)).squeeze()

#     def sample(self, N=1, alpha=2.5, tau=7.0, sigma=None, mean=None):
#         """
#         Use GaussianRF2d class for sampling (cleaner).
#         """
#         grf = GaussianRF(2, self.s1, alpha=2.5, tau=7, device=self.device)
#         u = grf.sample(N)
#         return u if N > 1 else u[0]


#     def plot_data(self, x, y, v, Jvp, file_path="plot.png", title="NS Sample Plot"):
#         """Plot input/output/eigenvector/JvP vorticity fields."""
#         def prep(f): return f.reshape(self.s1, self.s2).cpu().numpy()
#         fig, axs = plt.subplots(4, 1, figsize=(5, 20))
#         fig.suptitle(title)
#         for ax, field, label in zip(axs, [x, y, v, Jvp],
#                                     ["Input vorticity", "Output vorticity", "Eigenvector vorticity", "Jvp vorticity"]):
#             ax.imshow(prep(field), cmap="jet")
#             ax.set_title(label)
#         plt.savefig(file_path)
#         plt.close()

#     def plot_vorticity(self, w, step=0, file_path="vorticity.png", title="Vorticity Field"):
#         """
#         Thread-safe plotting of a single vorticity field.
#         """
#         fig, ax = plt.subplots(figsize=(6, 6))   # fresh figure+axes per call

#         im = ax.imshow(w,
#                     cmap="jet", origin="lower",
#                     extent=[0, 1., 0, 1.])
#         fig.colorbar(im, ax=ax, label="Vorticity", fraction=0.045, pad=0.06)

#         ax.set_title(title)
#         ax.set_xlabel("x")
#         ax.set_ylabel("y")

#         if file_path.endswith(".png"):
#             out_file = file_path.replace(".png", f"_{step}.png")
#         else:
#             out_file = f"{file_path}_{step}.png"

#         fig.savefig(out_file, dpi=150, bbox_inches="tight")
#         plt.close(fig)   # <-- close figure, important for threads

#     # Properties
#     @property
#     def domain(self):
#         return self.s1 * self.s2

#     @property
#     def range(self):
#         return self.s1 * self.s2
