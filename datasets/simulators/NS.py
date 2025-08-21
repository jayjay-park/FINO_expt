import torch
import torch.autograd.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from .base import Simulator
from scipy.fft import fft2, ifft2, fftshift
import torch.fft as fft

class NavierStokesSimulator(torch.nn.Module):
    '''
    configuration is desigend to match with Li et al. (2022)
    '''
    def __init__(self, 
                 s1, 
                 s2,
                 scale=10.0,
                 T=1.0,
                 Re=100,
                 adaptive=False,
                 delta_t=1e-4,
                 nburn=10,
                 nsteps=1000):

        super().__init__()

        self.s1 = s1
        self.s2 = s2
        self.scale = scale
        self.T = T
        self.Re = Re
        self.adaptive = adaptive
        self.delta_t = delta_t
        self.nburn = nburn
        self.nsteps = nsteps

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float64

        # TODO: Move to config
        L1 = 2*math.pi
        L2 = 2*math.pi
        self.L1 = L1
        self.L2 = L2


        self.h = 1.0/max(s1, s2)

        t = torch.linspace(0, 1, s1 + 1, device=self.device)
        t = t[0:-1]
        X, Y = torch.meshgrid(t, t)
        self.f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))) # forcing function


        #Wavenumbers for first derivatives
        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.zeros((1,)),\
                                torch.arange(start=-s1//2 + 1, end=0, step=1)), 0)
        self.k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(self.dtype).to(self.device)


        freq_list2 = torch.cat((torch.arange(start=0, end=s2//2, step=1), torch.zeros((1,))), 0)
        self.k2 = freq_list2.view(1,-1).repeat(s1, 1).type(self.dtype).to(self.device)

        #Negative Laplacian
        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(self.dtype).to(self.device)

        freq_list2 = torch.arange(start=0, end=s2//2 + 1, step=1)
        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(self.dtype).to(self.device)

        self.G = ((4*math.pi**2)/(L1**2))*k1**2 + ((4*math.pi**2)/(L2**2))*k2**2

        #Inverse of negative Laplacian
        self.inv_lap = self.G.clone()
        self.inv_lap[0,0] = 1.0
        self.inv_lap = 1.0/self.inv_lap

        #Dealiasing mask using 2/3 rule
        self.dealias = (self.k1**2 + self.k2**2 <= 0.6*(0.25*s1**2 + 0.25*s2**2)).type(self.dtype).to(self.device)
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

    def advance(self, w):

        #Rescale Laplacian by Reynolds number
        GG = (1.0/self.Re)*self.G

        #Move to Fourier space
        w_h = fft.rfft2(w)

        if self.f is not None:
            f_h = fft.rfft2(self.f)
        else:
            f_h = None
        
        if self.adaptive:
            q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
            delta_t = self.time_step(q, v, self.f, self.Re)

        time  = 0.0
        #Advance solution in Fourier space
        while time < self.T:
            if time + delta_t > self.T:
                current_delta_t = self.T - time
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
            if self.adaptive:
                q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
                delta_t = self.time_step(q, v, self.f, self.Re)
        
        return fft.irfft2(w_h, s=(self.s1, self.s2))
        
    def sample(self):
        nx, ny = self.s1, self.s2
        # Generate white noise in the Fourier domain (complex numbers)
        noise = np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)

        # Generate grid of frequency indices
        kx = np.fft.fftfreq(nx)[:, None]  # Frequency indices for x-axis
        ky = np.fft.fftfreq(ny)[None, :]  # Frequency indices for y-axis

        # Compute the radial frequency (distance from the origin in the Fourier domain)
        k = np.sqrt(kx**2 + ky**2)

        # Power spectrum: scale controls the smoothness (higher scale = smoother field)
        power_spectrum = np.exp(-k**2 * self.scale**2)

        # Apply the power spectrum to the noise
        field_fourier = noise * np.sqrt(power_spectrum)

        # Inverse Fourier transform to get the field in spatial domain
        field = np.real(ifft2(field_fourier))

        # Normalize the field (optional)
        field -= np.mean(field)
        field /= np.std(field)

        w0 = torch.tensor(field).to(self.device)
        for i in range(self.nburn):
            w0 = self.advance(w0)
        return w0
    
    def forward(self, w):
        for i in range(self.nsteps):
            if (i + 1) % 100 == 0:
                print(f"NS Simulator Step {i + 1} of {self.nsteps}")
            w = self.advance(w)
        return w

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
            field = field.reshape(self.s1, self.s2)
            return {
                'vorticity': field.cpu().numpy()
            }
        
        # Prepare all fields
        x_data = prepare_field(x)
        y_data = prepare_field(y)
        v_data = prepare_field(v)
        jvp_data = prepare_field(Jvp)
        
        # Create figure and subplots
        fig, axs = plt.subplots(4, 1, figsize=(5, 20))
        fig.suptitle(title)
        
        # Data to plot with corresponding titles
        plot_data = [
            (0, x_data, ['Input vorticity']),
            (1, y_data, ['Output vorticity']),
            (2, v_data, ['Eigenvector vorticity']),
            (3, jvp_data, ['Jvp vorticity'])
        ]
        
        # Plot all data
        for row, data, titles in plot_data:
            axs[row].imshow(data['vorticity'], cmap='jet')
            axs[row].set_title(titles[0])
        
        plt.savefig(file_path)
        plt.close()

    @property
    def domain(self):
        return self.s1 * self.s2
    
    @property
    def range(self):
        return self.s1 * self.s2
