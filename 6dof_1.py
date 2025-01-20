import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
m = 3.18  # weight
Ix, Iy, Iz = 0.029618, 0.069585, 0.042503  # moments of inertia
J = np.diag([Ix, Iy, Iz])
g = 9.81
k_t, k_r = 0.0, 0.0  # drag coefficients

# Differential equations
def rigid_body_dynamics(t, state, forces, torques):
    x, y, z, dx, dy, dz, phi, theta, psi, p, q, r = state
    u_f, tau_phi, tau_theta, tau_psi = forces
    
    # Linear accelerations
    ddx = (1 / m) * ((np.cos(phi) * np.cos(theta) * np.sin(theta) * u_f) + np.sin(phi) * np.sin(psi) * u_f - k_t * dx)
    ddy = (1 / m) * ((np.cos(phi) * np.sin(theta) * np.sin(psi) - np.cos(psi) * np.sin(phi)) * u_f - k_t * dy)
    ddz = (1 / m) * (np.cos(phi) * np.cos(theta) * u_f - m * g - k_t * dz)

    # Angular accelerations
    dp = (1 / Ix) * (-k_r * p - q * r * (Iz - Iy) + tau_phi)
    dq = (1 / Iy) * (-k_r * q - r * p * (Ix - Iz) + tau_theta)
    dr = (1 / Iz) * (-k_r * r - p * q * (Iy - Ix) + tau_psi)

    # Euler angle rates
    dphi = p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r
    dtheta = np.cos(phi) * q - np.sin(phi) * r
    dpsi = (1 / np.cos(theta)) * (np.sin(phi) * q + np.cos(phi) * r)

    return [dx, dy, dz, ddx, ddy, ddz, dphi, dtheta, dpsi, dp, dq, dr]

# Initial conditions
initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # x, y, z, dx, dy, dz, phi, theta, psi, p, q, r
forces = [(m+1)*g, 0, 0, 0]  # thrust, torques

time_span = (0, 10)
time_eval = np.linspace(time_span[0], time_span[1], 100)

# Solve the ODE
solution = solve_ivp(rigid_body_dynamics, time_span, initial_state, t_eval=time_eval, args=(forces, forces))

# Extract solutions
x, y, z = solution.y[0], solution.y[1], solution.y[2]
dx, dy, dz = solution.y[3], solution.y[4], solution.y[5]
p, q, r = solution.y[9], solution.y[10], solution.y[11]
phi, theta, psi = solution.y[6], solution.y[7], solution.y[8]

# Plot results
fig1, axs = plt.subplots(3, 1, figsize=(10, 8))
axs[0].plot(time_eval, x, label='x')
axs[0].plot(time_eval, y, label='y')
axs[0].plot(time_eval, z, label='z')
axs[0].set_title('Position over time')
axs[0].legend()

axs[1].plot(time_eval, dx, label='dx')
axs[1].plot(time_eval, dy, label='dy')
axs[1].plot(time_eval, dz, label='dz')
axs[1].set_title('Velocity over time')
axs[1].legend()

axs[2].plot(time_eval, phi, label='phi')
axs[2].plot(time_eval, theta, label='theta')
axs[2].plot(time_eval, psi, label='psi')
axs[2].set_title('Euler angles over time')
axs[2].legend()

plt.tight_layout()
plt.show()