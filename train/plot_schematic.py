# Let's update the plot:
# - Remove delta-space panel
# - Keep only GD and NGD (FINO) methods
# - Use stronger contour colors
# - FINO = indigo purple, GD = coral pink
# - Add iteration numbers and arrows
# - Increase gap between panels so arrow doesn't overlap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# --- Parameters ---
eta = 0.5
steps = 8

# Ill-conditioned Fisher in theta-space
I_theta = np.array([[1.0, 0.8],
                    [0.8, 2.5]])

# Start point
theta0 = np.array([2.5, 2.0])
optimum = np.array([0.0, 0.0])

# Perturbation for almost satisfied case
E = np.array([[0.05, 0.02],
              [0.02, 0.03]])
I_theta_perturbed = I_theta + E

# Cholesky for gamma transformation
L = np.linalg.cholesky(I_theta)
Linv = np.linalg.inv(L)

# --- Methods ---
def gd(theta0, A, steps, eta):
    theta = theta0.copy()
    traj = [theta.copy()]
    for _ in range(steps):
        grad = A @ theta
        theta = theta - eta * grad
        traj.append(theta.copy())
    return np.array(traj)

def ngd(theta0, A, I_theta, steps, eta):
    theta = theta0.copy()
    traj = [theta.copy()]
    I_inv = np.linalg.inv(I_theta)
    for _ in range(steps):
        grad = A @ theta
        theta = theta - eta * (I_inv @ grad)
        traj.append(theta.copy())
    return np.array(traj)

# --- Trajectories: perfect match ---
traj_gd_theta = gd(theta0, I_theta, steps, eta)
traj_ngd_theta = ngd(theta0, I_theta, I_theta, steps, eta)

# --- Trajectories: almost satisfied ---
traj_gd_theta_dash = gd(theta0, I_theta, steps, eta)
traj_ngd_theta_dash = ngd(theta0, I_theta, I_theta_perturbed, steps, eta)

# --- Gamma-space trajectories ---
traj_gd_gamma = (Linv @ traj_gd_theta.T).T
traj_ngd_gamma = (Linv @ traj_ngd_theta.T).T

traj_gd_gamma_dash = (Linv @ traj_gd_theta_dash.T).T
traj_ngd_gamma_dash = (Linv @ traj_ngd_theta_dash.T).T

# --- Contour data ---
def contour_data(A, xlim=(-3, 3), ylim=(-3, 3), n=200):
    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (A[0,0]*X**2 + 2*A[0,1]*X*Y + A[1,1]*Y**2)
    return X, Y, Z

X_theta, Y_theta, Z_theta = contour_data(I_theta, (-3, 3), (-3, 3))
X_gamma, Y_gamma, Z_gamma = contour_data(np.eye(2), (-3, 3), (-3, 3))

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors = {'gd': '#FF6F61',  # Coral pink
          'ngd': '#4B0082'} # Indigo purple

def plot_traj(ax, traj, traj_dash, label, color, marker):
    # Solid line
    ax.plot(traj[:,0], traj[:,1], marker+'-', color=color, label=label, lw=2, markersize=6)
    # Dashed line (almost satisfied)
    ax.plot(traj_dash[:,0], traj_dash[:,1], marker+'--', color=color, alpha=0.7, lw=2, markersize=6)
    # Iteration numbers
    for i, (x, y) in enumerate(traj):
        ax.text(x+0.05, y+0.05, str(i), fontsize=8, color=color)

# Theta-space
ax = axes[0]
ax.contour(X_theta, Y_theta, Z_theta, levels=15, cmap='plasma', alpha=0.8)
plot_traj(ax, traj_gd_theta, traj_gd_theta_dash, 'Numerical Simulator', colors['gd'], 'o')
plot_traj(ax, traj_ngd_theta, traj_ngd_theta_dash, 'FINO', colors['ngd'], 's')
ax.scatter(*theta0, color='k', marker='D', label='Start')
ax.scatter(*optimum, color='k', marker='X', s=60, label='Optimum')
ax.set_title(r'$\theta$-space', fontsize=16)
ax.set_xlabel(r'$\theta_1$', fontsize=14)
ax.set_ylabel(r'$\theta_2$', fontsize=14)
ax.grid(True)
ax.set_aspect('equal')

# Gamma-space
ax = axes[1]
ax.contour(X_gamma, Y_gamma, Z_gamma, levels=15, cmap='plasma', alpha=0.8)
plot_traj(ax, traj_gd_gamma, traj_gd_gamma_dash, 'Numerical Simulator', colors['gd'], 'o')
plot_traj(ax, traj_ngd_gamma, traj_ngd_gamma_dash, 'FINO', colors['ngd'], 's')
ax.scatter(*(Linv @ theta0), color='k', marker='D', label='Start')
ax.scatter(*optimum, color='k', marker='X', s=60, label='Optimum')
ax.set_title(r'$\gamma$-space ($\mathcal{I}_\gamma \approx I$)', fontsize=16)
ax.set_xlabel(r'$e_1$', fontsize=14)
ax.set_ylabel(r'$e_2$', fontsize=14)
ax.grid(True)
ax.set_aspect('equal')

# Legends
axes[0].legend(fontsize=14)

# Transformation arrow with more gap
def add_arrow(fig, ax1, ax2, text, gap=0.05):
    bbox1 = ax1.get_position()
    bbox2 = ax2.get_position()
    x0 = bbox1.x1 + gap
    x1 = bbox2.x0 - gap
    y = (bbox1.y0 + bbox1.y1) / 2
    arrow = FancyArrowPatch((x0, y), (x1, y),
                            transform=fig.transFigure,
                            arrowstyle='<->', mutation_scale=15, lw=1.5, color='black')
    fig.patches.append(arrow)
    fig.text((x0 + x1) / 2, y + 0.02, text,
             ha='center', va='center', fontsize=12)

add_arrow(fig, axes[0], axes[1], r'$\gamma = \zeta(\theta)$', gap=0.10)

plt.tight_layout()
plt.savefig("schematic.png")