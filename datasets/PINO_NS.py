import torch
import torch.fft as fft
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.fft import fft2, ifft2, fftshift


torch.manual_seed(0)


class GaussianRF(object):
    def __init__(self, dim, size, length=1.0, alpha=2.0, tau=3.0, sigma=None, boundary="periodic", constant_eig=False, device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        const = (4*(math.pi**2))/(length**2)

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((const*(k**2) + tau**2)**(-alpha/2.0))

            if constant_eig:
                self.sqrt_eig[0] = size*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((const*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))

            if constant_eig:
                self.sqrt_eig[0,0] = (size**2)*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((const*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))

            if constant_eig:
                self.sqrt_eig[0,0,0] = (size**3)*sigma*(tau**(-alpha))
            else:
                self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig*coeff

        u = torch.fft.irfftn(coeff, self.size, norm="backward")
        return u


class GaussianRF2d(object):

    def __init__(self, s1, s2, L1=2*math.pi, L2=2*math.pi, alpha=2.0, tau=3.0, sigma=None, mean=None, boundary="periodic", device=None, dtype=torch.float64):

        self.s1 = s1
        self.s2 = s2

        self.mean = mean

        self.device = device
        self.dtype = dtype

        if sigma is None:
            self.sigma = tau**(0.5*(2*alpha - 2.0))
        else:
            self.sigma = sigma

        const1 = (4*(math.pi**2))/(L1**2)
        const2 = (4*(math.pi**2))/(L2**2)

        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)

        freq_list2 = torch.arange(start=0, end=s2//2 + 1, step=1)

        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.sqrt_eig = s1*s2*self.sigma*((const1*k1**2 + const2*k2**2 + tau**2)**(-alpha/2.0))
        self.sqrt_eig[0,0] = 0.0

    def sample(self, N, xi=None):
        if xi is None:
            xi  = torch.randn(N, self.s1, self.s2//2 + 1, 2, dtype=self.dtype, device=self.device)
        
        xi[...,0] = self.sqrt_eig*xi [...,0]
        xi[...,1] = self.sqrt_eig*xi [...,1]
        
        u = torch.fft.irfft2(torch.view_as_complex(xi), s=(self.s1, self.s2))

        if self.mean is not None:
            u += self.mean
        
        return u

def gaussian_random_field_2d(size, scale=10.0, random_seed=None):
    """
    Generate a 2D Gaussian random field using spectral synthesis.

    Args:
    - size (tuple): Grid size (nx, ny)
    - scale (float): Controls the correlation length (larger values = smoother field)
    - random_seed (int): Random seed for reproducibility

    Returns:
    - field (2D numpy array): Generated Gaussian random field
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    nx, ny = size
    # Generate white noise in the Fourier domain (complex numbers)
    noise = np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)

    # Generate grid of frequency indices
    kx = np.fft.fftfreq(nx)[:, None]  # Frequency indices for x-axis
    ky = np.fft.fftfreq(ny)[None, :]  # Frequency indices for y-axis

    # Compute the radial frequency (distance from the origin in the Fourier domain)
    k = np.sqrt(kx**2 + ky**2)

    # Power spectrum: scale controls the smoothness (higher scale = smoother field)
    power_spectrum = np.exp(-k**2 * scale**2)

    # Apply the power spectrum to the noise
    field_fourier = noise * np.sqrt(power_spectrum)

    # Inverse Fourier transform to get the field in spatial domain
    field = np.real(ifft2(field_fourier))

    # Normalize the field (optional)
    field -= np.mean(field)
    field /= np.std(field)

    return torch.tensor(field)

#Setup for indexing in the 'ij' format

#Solve: -Lap(u) = f
class Poisson2d(object):

    def __init__(self, s1, s2, L1=2*math.pi, L2=2*math.pi, device=None, dtype=torch.float64):

        self.s1 = s1
        self.s2 = s2

        #Inverse negative Laplacian
        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)

        freq_list2 = torch.arange(start=0, end=s2//2 + 1, step=1)
        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.inv_lap = ((4*math.pi**2)/(L1**2))*k1**2 + ((4*math.pi**2)/(L2**2))*k2**2
        self.inv_lap[0,0] = 1.0
        self.inv_lap = 1.0/self.inv_lap
    
    def solve(self, f):
        return fft.irfft2(fft.rfft2(f)*self.inv_lap, s=(self.s1, self.s2))
    
    def __call__(self, f):
        return self.solve(f)

#Solve: w_t = - u . grad(w) + (1/Re)*Lap(w) + f
#       u = (psi_y, -psi_x)
#       -Lap(psi) = w
# Note: Adaptive time-step takes smallest step across the batch
# One step prediction
class NavierStokes2d(object):

    def __init__(self, s1, s2, L1=2*math.pi, L2=2*math.pi, device=None, dtype=torch.float64):

        self.s1 = s1
        self.s2 = s2

        self.L1 = L1
        self.L2 = L2

        self.h = 1.0/max(s1, s2)

        #Wavenumbers for first derivatives
        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.zeros((1,)),\
                                torch.arange(start=-s1//2 + 1, end=0, step=1)), 0)
        self.k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)


        freq_list2 = torch.cat((torch.arange(start=0, end=s2//2, step=1), torch.zeros((1,))), 0)
        self.k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        #Negative Laplacian
        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)

        freq_list2 = torch.arange(start=0, end=s2//2 + 1, step=1)
        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.G = ((4*math.pi**2)/(L1**2))*k1**2 + ((4*math.pi**2)/(L2**2))*k2**2

        #Inverse of negative Laplacian
        self.inv_lap = self.G.clone()
        self.inv_lap[0,0] = 1.0
        self.inv_lap = 1.0/self.inv_lap

        #Dealiasing mask using 2/3 rule
        self.dealias = (self.k1**2 + self.k2**2 <= 0.6*(0.25*s1**2 + 0.25*s2**2)).type(dtype).to(device)
        #Ensure mean zero
        self.dealias[0,0] = 0.0

    #Compute stream function from vorticity (Fourier space)
    def stream_function(self, w_h, real_space=False):
        #-Lap(psi) = w
        psi_h = self.inv_lap*w_h

        if real_space:
            return fft.irfft2(psi_h, s=(self.s1, self.s2))
        else:
            return psi_h

    #Compute velocity field from stream function (Fourier space)
    def velocity_field(self, stream_f, real_space=True):
        #Velocity field in x-direction = psi_y
        q_h = (2*math.pi/self.L2)*1j*self.k2*stream_f

        #Velocity field in y-direction = -psi_x
        v_h = -(2*math.pi/self.L1)*1j*self.k1*stream_f

        if real_space:
            return fft.irfft2(q_h, s=(self.s1, self.s2)), fft.irfft2(v_h, s=(self.s1, self.s2))
        else:
            return q_h, v_h

    #Compute non-linear term + forcing from given vorticity (Fourier space)
    def nonlinear_term(self, w_h, f_h=None):
        #Physical space vorticity
        w = fft.irfft2(w_h, s=(self.s1, self.s2))

        #Physical space velocity
        q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)

        #Compute non-linear term in Fourier space
        nonlin = -1j*((2*math.pi/self.L1)*self.k1*fft.rfft2(q*w) + (2*math.pi/self.L1)*self.k2*fft.rfft2(v*w))

        #Add forcing function
        if f_h is not None:
            nonlin += f_h

        return nonlin
    
    def time_step(self, q, v, f, Re):
        #Maxixum speed
        max_speed = torch.max(torch.sqrt(q**2 + v**2)).item()

        #Maximum force amplitude
        if f is not None:
            xi = torch.sqrt(torch.max(torch.abs(f))).item()
        else:
            xi = 1.0
        
        #Viscosity
        mu = (1.0/Re)*xi*((self.L1/(2*math.pi))**(3.0/4.0))*(((self.L2/(2*math.pi))**(3.0/4.0)))

        if max_speed == 0:
            return 0.5*(self.h**2)/mu
        
        #Time step based on CFL condition
        return min(0.5*self.h/max_speed, 0.5*(self.h**2)/mu)

    def advance(self, w, f=None, T=1.0, Re=100, adaptive=True, delta_t=1e-3):

        #Rescale Laplacian by Reynolds number
        GG = (1.0/Re)*self.G

        #Move to Fourier space
        w_h = fft.rfft2(w)

        if f is not None:
            f_h = fft.rfft2(f)
        else:
            f_h = None
        
        if adaptive:
            q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
            delta_t = self.time_step(q, v, f, Re)

        time  = 0.0
        #Advance solution in Fourier space
        while time < T:
            if time + delta_t > T:
                current_delta_t = T - time
            else:
                current_delta_t = delta_t

            #Inner-step of Heun's method
            nonlin1 = self.nonlinear_term(w_h, f_h)
            w_h_tilde = (w_h + current_delta_t*(nonlin1 - 0.5*GG*w_h))/(1.0 + 0.5*current_delta_t*GG)

            #Cranck-Nicholson + Heun update
            nonlin2 = self.nonlinear_term(w_h_tilde, f_h)
            w_h = (w_h + current_delta_t*(0.5*(nonlin1 + nonlin2) - 0.5*GG*w_h))/(1.0 + 0.5*current_delta_t*GG)

            #De-alias
            w_h *= self.dealias

            #Update time
            time += current_delta_t

            #New time step
            if adaptive:
                q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
                delta_t = self.time_step(q, v, f, Re)
        
        return fft.irfft2(w_h, s=(self.s1, self.s2))
    
    def __call__(self, w, f=None, T=1.0, Re=100, adaptive=True, delta_t=1e-3):
        return self.advance(w, f, T, Re, adaptive, delta_t)

# Function to plot vorticity
def plot_vorticity(vorticity_field, i, title="Vorticity Field"):
    plt.figure(figsize=(6, 6))
    plt.imshow(vorticity_field, cmap='jet', origin='lower', extent=[0, 1., 0, 1.]) #[0, 2*math.pi, 0, 2*math.pi]
    plt.colorbar(label='Vorticity', fraction=0.045, pad=0.06)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'PINO_NS/data_new_vorticity_{i}.png')
    plt.close()

if __name__ == "__main__":
    # Parameters
    s1, s2 = 64, 64  # Grid size
    L1, L2 = 2*math.pi, 2*math.pi  # Domain size
    Re = 100 #200  # Reynolds number
    T = 10.0  # Simulation time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Navier-Stokes solver
    ns_solver = NavierStokes2d(s1, s2, L1=L1, L2=L2, device=device)

    # Generate initial vorticity (random or sinusoidal)
    # # Set up 2d GRF with covariance parameters
    GRF = GaussianRF(2, s1, alpha=2.5, tau=7, device=device)
    # # w = torch.randn(s1, s2, device=device, dtype=torch.float64)
    bsize=1
    w = GRF.sample(bsize).cuda().squeeze()

    # Optionally, define a forcing function (or set to None)
    t = torch.linspace(0, 1, s1 + 1, device=device)
    t = t[0:-1]

    X, Y = torch.meshgrid(t, t)
    f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

    # Run the simulation to get the vorticity at time T
    # w_t_final = ns_solver(w, f=f, T=T, Re=Re)
    # plot_vorticity(w_t_final.squeeze().detach().cpu(), 1500)

    # Initial condition (requires grad not needed for jvp)
    w0 = w.clone()

    # Direction vector (perturbation)
    v = torch.randn_like(w0)
    v1 = torch.randn_like(w0)

    # Define function: advance by T=10 * Δt
    fwd = lambda w: ns_solver(w, f=f, T=T, Re=Re)

    # Compute forward state and Jv
    w10, jvp_val = torch.func.jvp(fwd, (w0,), (v,))
    plot_vorticity(w10.squeeze().detach().cpu(), 2000)
    plot_vorticity(jvp_val.squeeze().detach().cpu(), 3000)

    print("w10 shape:", w10.shape)       # (s1, s2)
    print("Jv shape :", jvp_val.shape)   # (s1, s2)


    # # If you want to generate multiple time steps and store them, you can modify the code like this:
    # vorticity_data = []

    # w_current = w
    # for i in range(num_time_steps):
    #     w_current = ns_solver(w_current, f=f, T=time_step, Re=Re)
    #     vorticity_data.append(w_current.cpu().numpy())  # Store the result

    # print("vorticity data shape", torch.tensor(vorticity_data).shape)

    # # To plot intermediate vorticity fields from vorticity_data (generated for multiple steps)
    # for i, vorticity_step in enumerate(vorticity_data):
    #     if i % 10 == 0:
    #         plot_vorticity(vorticity_step.squeeze(), i, title=f"Vorticity Field at Time Step {i + 1}")
