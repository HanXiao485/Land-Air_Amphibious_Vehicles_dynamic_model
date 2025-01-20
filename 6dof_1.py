import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Line3D

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
forces = [(m+0.1)*g, 0, 0.001, 0.001]  # thrust, torques

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

# Animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(0, 10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Drone Trajectory Visualization")

trajectory_line, = ax.plot([], [], [], 'b-', label="Trajectory")
initial_axes = []
real_time_axes = []

# Add initial and real-time orientation axes
def draw_axes(center, R, length=1, alpha=0.8):
    colors = ['r', 'g', 'b']  # x, y, z
    axes = []
    for i in range(3):
        start = center
        end = center + length * R[:, i]
        axes.append(Line3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=colors[i], alpha=alpha))
        ax.add_line(axes[-1])
    return axes

# Initialize animation
def init():
    trajectory_line.set_data([], [])
    trajectory_line.set_3d_properties([])
    R_initial = np.eye(3)  # Identity matrix for initial orientation
    global initial_axes
    initial_axes = draw_axes(np.array([x[0], y[0], z[0]]), R_initial)
    return trajectory_line, *initial_axes

# Update animation
def update(frame):
    trajectory_line.set_data(x[:frame], y[:frame])
    trajectory_line.set_3d_properties(z[:frame])

    # Remove previous real-time axes
    global real_time_axes
    for line in real_time_axes:
        line.remove()
    real_time_axes = []

    # Calculate rotation matrix
    phi_f, theta_f, psi_f = phi[frame], theta[frame], psi[frame]
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(phi_f), -np.sin(phi_f)],
                    [0, np.sin(phi_f), np.cos(phi_f)]])
    R_y = np.array([[np.cos(theta_f), 0, np.sin(theta_f)],
                    [0, 1, 0],
                    [-np.sin(theta_f), 0, np.cos(theta_f)]])
    R_z = np.array([[np.cos(psi_f), -np.sin(psi_f), 0],
                    [np.sin(psi_f), np.cos(psi_f), 0],
                    [0, 0, 1]])
    R = R_z @ R_y @ R_x
    center = np.array([x[frame], y[frame], z[frame]])
    real_time_axes = draw_axes(center, R, alpha=0.9)
    return trajectory_line, *real_time_axes

ani = FuncAnimation(fig, update, frames=len(time_eval), init_func=init, blit=False, interval=50)

plt.legend()
plt.show()