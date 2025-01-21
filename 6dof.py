import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D

class DroneSimulation:
    def __init__(self, mass=3.18, inertia=(0.029618, 0.069585, 0.042503), drag_coeffs=(0.0, 0.0), gravity=9.81):
        """
        Initialize the drone simulation with given parameters.

        Parameters:
        - mass: Mass of the drone (kg)
        - inertia: Tuple containing moments of inertia (Ix, Iy, Iz)
        - drag_coeffs: Tuple containing linear and angular drag coefficients (k_t, k_r)
        - gravity: Gravitational acceleration (m/s^2)
        """
        self.m = mass
        self.Ix, self.Iy, self.Iz = inertia
        self.J = np.diag([self.Ix, self.Iy, self.Iz])
        self.k_t, self.k_r = drag_coeffs
        self.g = gravity

    def rigid_body_dynamics(self, t, state, forces):
        x, y, z, dx, dy, dz, phi, theta, psi, p, q, r = state
        u_f, tau_phi, tau_theta, tau_psi = forces

        # Linear accelerations
        ddx = (1 / self.m) * ((np.cos(phi) * np.cos(theta) * np.sin(theta) * u_f) + np.sin(phi) * np.sin(psi) * u_f - self.k_t * dx)
        ddy = (1 / self.m) * ((np.cos(phi) * np.sin(theta) * np.sin(psi) - np.cos(psi) * np.sin(phi)) * u_f - self.k_t * dy)
        ddz = (1 / self.m) * (np.cos(phi) * np.cos(theta) * u_f - self.m * self.g - self.k_t * dz)

        # Angular accelerations
        dp = (1 / self.Ix) * (-self.k_r * p - q * r * (self.Iz - self.Iy) + tau_phi)
        dq = (1 / self.Iy) * (-self.k_r * q - r * p * (self.Ix - self.Iz) + tau_theta)
        dr = (1 / self.Iz) * (-self.k_r * r - p * q * (self.Iy - self.Ix) + tau_psi)

        # Euler angle rates
        dphi = p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r
        dtheta = np.cos(phi) * q - np.sin(phi) * r
        dpsi = (1 / np.cos(theta)) * (np.sin(phi) * q + np.cos(phi) * r)

        return [dx, dy, dz, ddx, ddy, ddz, dphi, dtheta, dpsi, dp, dq, dr]

    def simulate(self, initial_state, forces, time_span, time_eval):
        """
        Run the simulation for the given initial state, forces, and time range.

        Parameters:
        - initial_state: List of initial conditions [x, y, z, dx, dy, dz, phi, theta, psi, p, q, r]
        - forces: List of forces and torques [u_f, tau_phi, tau_theta, tau_psi]
        - time_span: Tuple indicating start and end time (t0, tf)
        - time_eval: Array of time points for evaluation
        """
        self.solution = solve_ivp(self.rigid_body_dynamics, time_span, initial_state, t_eval=time_eval, args=(forces,))
        self.time_eval = time_eval

    def plot_results(self):
        """Plot the simulation results."""
        solution = self.solution
        x, y, z = solution.y[0], solution.y[1], solution.y[2]
        dx, dy, dz = solution.y[3], solution.y[4], solution.y[5]
        phi, theta, psi = solution.y[6], solution.y[7], solution.y[8]

        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        axs[0].plot(self.time_eval, x, label='x')
        axs[0].plot(self.time_eval, y, label='y')
        axs[0].plot(self.time_eval, z, label='z')
        axs[0].set_title('Position over time')
        axs[0].legend()

        axs[1].plot(self.time_eval, dx, label='dx')
        axs[1].plot(self.time_eval, dy, label='dy')
        axs[1].plot(self.time_eval, dz, label='dz')
        axs[1].set_title('Velocity over time')
        axs[1].legend()

        axs[2].plot(self.time_eval, phi, label='phi')
        axs[2].plot(self.time_eval, theta, label='theta')
        axs[2].plot(self.time_eval, psi, label='psi')
        axs[2].set_title('Euler angles over time')
        axs[2].legend()

        plt.tight_layout()
        plt.show()

    def animate_trajectory(self):
        """Create an animation of the drone's trajectory."""
        solution = self.solution
        x, y, z = solution.y[0], solution.y[1], solution.y[2]
        phi, theta, psi = solution.y[6], solution.y[7], solution.y[8]

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
        real_time_axes = []

        def draw_axes(center, R, length=1, alpha=0.8):
            colors = ['r', 'g', 'b']  # x, y, z
            axes = []
            for i in range(3):
                start = center
                end = center + length * R[:, i]
                axes.append(Line3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=colors[i], alpha=alpha))
                ax.add_line(axes[-1])
            return axes
        
        # Draw the static (initial) coordinate system
        initial_position = np.array([x[0], y[0], z[0]])  # The initial position
        initial_phi, initial_theta, initial_psi = phi[0], theta[0], psi[0]  # The initial Euler angles
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(initial_phi), -np.sin(initial_phi)],
                        [0, np.sin(initial_phi), np.cos(initial_phi)]])
        R_y = np.array([[np.cos(initial_theta), 0, np.sin(initial_theta)],
                        [0, 1, 0],
                        [-np.sin(initial_theta), 0, np.cos(initial_theta)]])
        R_z = np.array([[np.cos(initial_psi), -np.sin(initial_psi), 0],
                        [np.sin(initial_psi), np.cos(initial_psi), 0],
                        [0, 0, 1]])
        R_initial = R_z @ R_y @ R_x
        draw_axes(initial_position, R_initial, alpha=1.0)  # Fixed coordinate system at the initial position

        def init():
            trajectory_line.set_data([], [])
            trajectory_line.set_3d_properties([])
            return trajectory_line

        def update(frame):
            trajectory_line.set_data(x[:frame], y[:frame])
            trajectory_line.set_3d_properties(z[:frame])

            nonlocal real_time_axes
            for line in real_time_axes:
                line.remove()
            real_time_axes = []

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

        ani = FuncAnimation(fig, update, frames=len(self.time_eval), init_func=init, blit=False, interval=50)
        plt.legend()
        plt.show()


# Test the DroneSimulation class
def main():
    # Initialize the simulation
    drone = DroneSimulation()

    # Initial conditions: [x, y, z, dx, dy, dz, phi, theta, psi, p, q, r]
    initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Forces and torques: [u_f, tau_phi, tau_theta, tau_psi]
    forces = [(3.18+0.1)*9.81, 0.0, 0.001, 0.0]

    # Time span and evaluation points
    time_span = (0, 10)
    time_eval = np.linspace(0, 10, 100)

    # Simulate the dynamics
    drone.simulate(initial_state, forces, time_span, time_eval)

    # Plot the results
    drone.plot_results()

    # Animate the trajectory
    drone.animate_trajectory()


if __name__ == "__main__":
    main()