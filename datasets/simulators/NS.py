import torch
import torch.fft as fft
import torch.utils.checkpoint as checkpoint
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.fft import ifft2


torch.manual_seed(0)


# =====================================================
# Gaussian Random Field Generator
# =====================================================
class GaussianRF(object):
    def __init__(self, dim, size, length=1.0, alpha=2.0, tau=3.0,
                 sigma=None, boundary="periodic", constant_eig=False, device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5 * (2 * alpha - self.dim))

        k_max = size // 2
        const = (4 * (math.pi**2)) / (length**2)

        if dim == 2:
            wavenumbers = torch.cat(
                (torch.arange(0, k_max, device=device),
                 torch.arange(-k_max, 0, device=device)), 0).repeat(size, 1)
            k_x = wavenumbers.transpose(0, 1)
            k_y = wavenumbers
            self.sqrt_eig = (size**2) * math.sqrt(2.0) * sigma * \
                ((const * (k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0, 0] = 0.0

        self.size = (size, size)

    def sample(self, N=1):
        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff
        u = torch.fft.irfftn(coeff, self.size, norm="backward")
        return u


# =====================================================
# Navier–Stokes Spectral Solver
# =====================================================
class NavierStokes2d(torch.nn.Module):
    def __init__(self, s1, s2, L1=2*math.pi, L2=2*math.pi,
                 Re=100, adaptive=False, delta_t=1e-3,
                 nburn=10, nsteps=1000, device=None, T=10, dtype=torch.float64):
        super().__init__()

        self.s1, self.s2 = s1, s2
        self.L1, self.L2 = L1, L2
        self.Re = Re
        self.adaptive = adaptive
        self.delta_t = delta_t
        self.nburn = nburn
        self.nsteps = nsteps
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.T = T

        self.h = 1.0 / max(s1, s2)

        # Wavenumbers
        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),
                                torch.zeros((1,)),
                                torch.arange(start=-s1//2 + 1, end=0, step=1)), 0)
        self.k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(self.device)

        freq_list2 = torch.cat((torch.arange(start=0, end=s2//2, step=1),
                                torch.zeros((1,))), 0)
        self.k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(self.device)

        # Negative Laplacian
        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(self.device)

        freq_list2 = torch.arange(start=0, end=s2//2 + 1, step=1)
        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(self.device)

        self.G = ((4*math.pi**2)/(L1**2)) * k1**2 + ((4*math.pi**2)/(L2**2)) * k2**2
        self.inv_lap = 1.0 / self.G.clone()
        self.inv_lap[0, 0] = 0.0

        # Dealias mask
        self.dealias = (k1**2 + k2**2 <= 0.6 * (0.25 * s1**2 + 0.25 * s2**2)).type(dtype)
        self.dealias[0, 0] = 0.0

        # Forcing
        t = torch.linspace(0, 1, s1 + 1, device=self.device)[:-1]
        X, Y = torch.meshgrid(t, t, indexing="ij")
        self.f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

    # Stream function
    def stream_function(self, w_h, real_space=False):
        psi_h = self.inv_lap * w_h
        return fft.irfft2(psi_h, s=(self.s1, self.s2)) if real_space else psi_h

    # Velocity field
    def velocity_field(self, stream_f, real_space=True):
        freq_k1 = torch.fft.fftfreq(self.s1, d=1/self.s1).to(self.device)
        freq_k2 = torch.fft.rfftfreq(self.s2, d=1/self.s2).to(self.device)
        k1 = freq_k1.view(-1, 1).repeat(1, self.s2 // 2 + 1)
        k2 = freq_k2.view(1, -1).repeat(self.s1, 1)
        q_h = (2 * math.pi / self.L2) * 1j * k2 * stream_f
        v_h = -(2 * math.pi / self.L1) * 1j * k1 * stream_f
        return (fft.irfft2(q_h, s=(self.s1, self.s2)),
                fft.irfft2(v_h, s=(self.s1, self.s2))) if real_space else (q_h, v_h)

    # Nonlinear term
    def nonlinear_term(self, w_h, f_h=None):
        w = fft.irfft2(w_h, s=(self.s1, self.s2))
        q, v = self.velocity_field(self.stream_function(w_h), real_space=True)
        nonlin = -1j * ((2 * math.pi / self.L1) * self.k1 * fft.rfft2(q * w) +
                        (2 * math.pi / self.L2) * self.k2 * fft.rfft2(v * w))
        if f_h is not None:
            nonlin += f_h
        return nonlin

    # Advance solution
    def advance(self, w):
        GG = (1.0 / self.Re) * self.G
        w_h = fft.rfft2(w)
        f_h = fft.rfft2(self.f) if self.f is not None else None

        delta_t = self.delta_t
        if self.adaptive:
            q, v = self.velocity_field(self.stream_function(w_h))
            delta_t = self.time_step(q, v, self.f)

        time = 0.0
        while time < self.T:
            current_delta_t = min(delta_t, self.T - time)
            nonlin1 = self.nonlinear_term(w_h, f_h)
            w_h_tilde = (w_h + current_delta_t * (nonlin1 - 0.5 * GG * w_h)) / (1.0 + 0.5 * current_delta_t * GG)
            nonlin2 = self.nonlinear_term(w_h_tilde, f_h)
            w_h = (w_h + current_delta_t * (0.5 * (nonlin1 + nonlin2) - 0.5 * GG * w_h)) / (1.0 + 0.5 * current_delta_t * GG)
            w_h *= self.dealias
            time += current_delta_t
        return fft.irfft2(w_h, s=(self.s1, self.s2))

    # Forward rollout with checkpointing
    def forward(self, w):
        for i in range(self.nsteps):
            # w = checkpoint.checkpoint(self.advance, w)
            w = self.advance(w)
            if (i + 1) % 1 == 0:
                print(f"NS Simulator Step {i + 1}/{self.nsteps}")
                self.plot_vorticity(w.squeeze().detach().cpu(),
                       i=i,
                       title=f"Vorticity Field at Step {i}")
        return w


    # =====================================================
    # Utility: Plot vorticity
    # =====================================================
    def plot_vorticity(self, vorticity_field, i, title="Vorticity Field"):
        plt.figure(figsize=(6, 6))
        plt.imshow(vorticity_field, cmap='jet', origin='lower')
        plt.colorbar(label='Vorticity')
        plt.title(title)
        plt.savefig(f'../PINO_NS/NS_vorticity_{i}.png')
        plt.close()


# =====================================================
# Example Run
# =====================================================
if __name__ == "__main__":
    # --- Parameters ---
    s1, s2 = 128, 128          # grid size
    Re = 200                   # Reynolds number (FNO-style)
    T = 10.0                   # total simulation time
    nsteps = 1000              # number of time steps
    delta_t = T / nsteps     # fixed Δt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Initialize solver ---
    ns_solver = NavierStokes2d(
        s1, s2,
        Re=Re,
        T=T,
        nsteps=nsteps,
        delta_t=delta_t,
        device=device
    )
    GRF = GaussianRF(2, s1, alpha=2.5, tau=7, device=device)

    # --- Initial condition from GRF ---
    w0 = GRF.sample(1).to(device).squeeze()

    # --- Forcing term ---
    t = torch.linspace(0, 1, s1, device=device)
    X, Y = torch.meshgrid(t, t, indexing="ij")
    f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) +
               torch.cos(2 * math.pi * (X + Y)))

    # --- Rollout simulation ---
    vorticity_data = []
    w_final = ns_solver(w0)

    # for i in range(nsteps):
    #     w_current = ns_solver(w_current)

    #     # Store only every 10th snapshot to save memory
    #     if (i + 1) % 10 == 0:
    #         vorticity_data.append(w_current.detach().cpu().numpy())
    #         print(f"Stored vorticity at step {i+1}/{nsteps}")

    # vorticity_data = np.array(vorticity_data)  # [#snapshots, s1, s2]
    # print("vorticity data shape:", vorticity_data.shape)

    # # --- Plot stored snapshots ---
    # for i, vorticity_step in enumerate(vorticity_data):
    #     step_num = (i + 1) * 10
    #     plot_vorticity(vorticity_step.squeeze(),
    #                    i=step_num,
    #                    title=f"Vorticity Field at Step {step_num}")
