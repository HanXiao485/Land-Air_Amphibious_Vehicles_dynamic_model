import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D
import configparser
import csv
import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Read configuration file
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'config.ini')
config = configparser.ConfigParser()
config.read(file_path)

########################################################################
# Fourth-order Runge-Kutta Integrator Class (with callback)
########################################################################
class RK4Integrator:
    """
    Fourth-order Runge-Kutta integrator.
    After each integration step, calls a callback function to update the control input.
    """
    def __init__(self, func, forces):
        self.func = func
        self.forces = forces
        self.states = []

    def integrate(self, time_eval, initial_state, callback=None):
        dt = time_eval[1] - time_eval[0]
        state = np.array(initial_state)
        self.states = []
        for idx in range(len(time_eval) - 1):
            self.states.append(state.copy())
            t_current = time_eval[idx]
            k1 = np.array(self.func(t_current, state, self.forces))
            k2 = np.array(self.func(t_current + dt/2, state + dt/2 * k1, self.forces))
            k3 = np.array(self.func(t_current + dt/2, state + dt/2 * k2, self.forces))
            k4 = np.array(self.func(t_current + dt, state + dt * k3, self.forces))
            new_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            if new_state[2] < 0:
                new_state[2] = 0
                new_state[5] = 0
            if callback is not None:
                new_forces = callback(time_eval[idx+1], new_state, self.forces)
                if new_forces is not None:
                    self.forces = new_forces
            state = new_state
        self.states.append(state.copy())
        return time_eval, np.array(self.states)

########################################################################
# PID Controller Class with Three Loops (Position, Attitude, Rate)
########################################################################
class DualLoopPIDController:
    """
    Four-loop PID controller:
      - Outer loop (Position control): calculates desired acceleration based on position error,
        then computes the desired velocity.
      - Second loop (Velocity control): calculates desired velocity based on velocity error.
      - Middle loop (Attitude control): calculates desired angular rates based on attitude error.
      - Inner loop (Rate control): calculates control moments from angular rate errors via PID control.
      
      The total lift force is computed as:
          u_f = mass * (g + a_z_des)
    """
    def __init__(self, mass, gravity, desired_position, desired_velocity, desired_attitude, dt,
                 # Outer loop PID parameters (Position control)
                 kp_x=1.0, ki_x=0.0, kd_x=0.5,
                 kp_y=1.0, ki_y=0.0, kd_y=0.5,
                 kp_z=2.0, ki_z=0.0, kd_z=1.0,
                 # Second loop PID parameters (Velocity control)
                 kp_vx=1.0, ki_vx=0.0, kd_vx=0.5,
                 kp_vy=1.0, ki_vy=0.0, kd_vy=0.5,
                 kp_vz=2.0, ki_vz=0.0, kd_vz=1.0,
                 # Middle loop PID parameters (Attitude control)
                 att_kp_phi=5.0, att_ki_phi=0.1, att_kd_phi=2.0,
                 att_kp_theta=5.0, att_ki_theta=0.1, att_kd_theta=2.0,
                 att_kp_psi=1.0, att_ki_psi=0.0, att_kd_psi=0.2,
                 # Inner loop PID parameters (Angular rate control)
                 rate_kp_phi=2.0, rate_ki_phi=0.0, rate_kd_phi=0.5,
                 rate_kp_theta=2.0, rate_ki_theta=0.0, rate_kd_theta=0.5,
                 rate_kp_psi=1.0, rate_ki_psi=0.0, rate_kd_psi=0.2):
        self.mass = mass
        self.g = gravity
        self.desired_position = desired_position      # Target position: (x_des, y_des, z_des)
        self.desired_velocity = desired_velocity      # Target velocity: (vx_des, vy_des, vz_des)
        self.desired_attitude = desired_attitude      # Target attitude: (phi_des, theta_des, psi_des)
        self.dt = dt

        # Outer loop PID parameters (Position control)
        self.Kp_x = kp_x; self.Ki_x = ki_x; self.Kd_x = kd_x
        self.Kp_y = kp_y; self.Ki_y = ki_y; self.Kd_y = kd_y
        self.Kp_z = kp_z; self.Ki_z = ki_z; self.Kd_z = kd_z

        # Second loop PID parameters (Velocity control)
        self.Kp_vx = kp_vx; self.Ki_vx = ki_vx; self.Kd_vx = kd_vx
        self.Kp_vy = kp_vy; self.Ki_vy = ki_vy; self.Kd_vy = kd_vy
        self.Kp_vz = kp_vz; self.Ki_vz = ki_vz; self.Kd_vz = kd_vz

        # Middle loop PID parameters (Attitude control)
        self.att_kp_phi = att_kp_phi; self.att_ki_phi = att_ki_phi; self.att_kd_phi = att_kd_phi
        self.att_kp_theta = att_kp_theta; self.att_ki_theta = att_ki_theta; self.att_kd_theta = att_kd_theta
        self.att_kp_psi = att_kp_psi; self.att_ki_psi = att_ki_psi; self.att_kd_psi = att_kd_psi

        # Inner loop PID parameters (Angular rate control)
        self.rate_kp_phi = rate_kp_phi; self.rate_ki_phi = rate_ki_phi; self.rate_kd_phi = rate_kd_phi
        self.rate_kp_theta = rate_kp_theta; self.rate_ki_theta = rate_ki_theta; self.rate_kd_theta = rate_kd_theta
        self.rate_kp_psi = rate_kp_psi; self.rate_ki_psi = rate_ki_psi; self.rate_kd_psi = rate_kd_psi

        # Initialize integrations and previous errors
        self.int_x = 0.0; self.last_error_x = 0.0
        self.int_y = 0.0; self.last_error_y = 0.0
        self.int_z = 0.0; self.last_error_z = 0.0

        self.int_vx = 0.0; self.last_error_vx = 0.0
        self.int_vy = 0.0; self.last_error_vy = 0.0
        self.int_vz = 0.0; self.last_error_vz = 0.0

        self.int_phi_att = 0.0; self.last_error_phi_att = 0.0
        self.int_theta_att = 0.0; self.last_error_theta_att = 0.0
        self.int_psi_att = 0.0; self.last_error_psi_att = 0.0

        self.int_p = 0.0; self.last_error_p = 0.0
        self.int_q = 0.0; self.last_error_q = 0.0
        self.int_r = 0.0; self.last_error_r = 0.0

        self.last_time = None

    def update(self, current_time, state):
        """
        Compute new control input based on current time and state.
        State: [x, y, z, dx, dy, dz, phi, theta, psi, p, q, r]
        Output: [lift_force, tau_phi, tau_theta, tau_psi]
        """
        if self.last_time is None:
            dt = self.dt
        else:
            dt = current_time - self.last_time
        self.last_time = current_time

        # Extract state variables
        x, y, z, dx, dy, dz, phi, theta, psi, p, q, r = state
        x_des, y_des, z_des = self.desired_position
        vx_des, vy_des, vz_des = self.desired_velocity

        # Outer loop: Position control
        error_x = x_des - x
        error_y = y_des - y
        error_z = z_des - z

        self.int_x += error_x * dt
        self.int_y += error_y * dt
        self.int_z += error_z * dt

        d_error_x = (error_x - self.last_error_x) / dt
        d_error_y = (error_y - self.last_error_y) / dt
        d_error_z = (error_z - self.last_error_z) / dt

        self.last_error_x = error_x
        self.last_error_y = error_y
        self.last_error_z = error_z

        # Compute desired velocity for the velocity control loop
        vx_des = self.Kp_x * error_x + self.Ki_x * self.int_x + self.Kd_x * d_error_x
        vy_des = self.Kp_y * error_y + self.Ki_y * self.int_y + self.Kd_y * d_error_y
        vz_des = self.Kp_z * error_z + self.Ki_z * self.int_z + self.Kd_z * d_error_z

        # Second loop: Velocity control
        error_vx = vx_des - dx
        error_vy = vy_des - dy
        error_vz = vz_des - dz

        self.int_vx += error_vx * dt
        self.int_vy += error_vy * dt
        self.int_vz += error_vz * dt

        d_error_vx = (error_vx - self.last_error_vx) / dt
        d_error_vy = (error_vy - self.last_error_vy) / dt
        d_error_vz = (error_vz - self.last_error_vz) / dt

        self.last_error_vx = error_vx
        self.last_error_vy = error_vy
        self.last_error_vz = error_vz
        
        ax_des = self.Kp_vx * error_vx + self.Ki_vx * self.int_vx + self.Kd_vx * d_error_vx
        ay_des = self.Kp_vy * error_vy + self.Ki_vy * self.int_vy + self.Kd_vy * d_error_vy
        az_des = self.Kp_vz * error_vz + self.Ki_vz * self.int_vz + self.Kd_vz * d_error_vz

        # Compute desired attitude based on velocity control
        phi_des_pos = (1.0 / self.g) * ay_des
        theta_des_pos = (1.0 / self.g) * ax_des

        if self.desired_attitude[0] == 0:
            phi_des = phi_des_pos
        else:
            phi_des = self.desired_attitude[0]
        if self.desired_attitude[1] == 0:
            theta_des = theta_des_pos
        else:
            theta_des = self.desired_attitude[1]
        psi_des = self.desired_attitude[2]

        # Middle loop: Attitude control
        error_phi = phi_des - phi
        error_theta = theta_des - theta
        error_psi = psi_des - psi

        self.int_phi_att += error_phi * dt
        self.int_theta_att += error_theta * dt
        self.int_psi_att += error_psi * dt

        d_error_phi_att = (error_phi - self.last_error_phi_att) / dt
        d_error_theta_att = (error_theta - self.last_error_theta_att) / dt
        d_error_psi_att = (error_psi - self.last_error_psi_att) / dt

        self.last_error_phi_att = error_phi
        self.last_error_theta_att = error_theta
        self.last_error_psi_att = error_psi

        p_des = self.att_kp_phi * error_phi + self.att_ki_phi * self.int_phi_att + self.att_kd_phi * d_error_phi_att
        q_des = self.att_kp_theta * error_theta + self.att_ki_theta * self.int_theta_att + self.att_kd_theta * d_error_theta_att
        r_des = self.att_kp_psi * error_psi + self.att_ki_psi * self.int_psi_att + self.att_kd_psi * d_error_psi_att

        # Inner loop: Angular rate control
        error_p = p_des - p
        error_q = q_des - q
        error_r = r_des - r

        self.int_p += error_p * dt
        self.int_q += error_q * dt
        self.int_r += error_r * dt

        d_error_p = (error_p - self.last_error_p) / dt
        d_error_q = (error_q - self.last_error_q) / dt
        d_error_r = (error_r - self.last_error_r) / dt

        self.last_error_p = error_p
        self.last_error_q = error_q
        self.last_error_r = error_r

        tau_phi = self.rate_kp_phi * error_p + self.rate_ki_phi * self.int_p + self.rate_kd_phi * d_error_p
        tau_theta = self.rate_kp_theta * error_q + self.rate_ki_theta * self.int_q + self.rate_kd_theta * d_error_q
        tau_psi = self.rate_kp_psi * error_r + self.rate_ki_psi * self.int_r + self.rate_kd_psi * d_error_r

        # Update the calculation of lift force to allow free fall when no control is set
        u_f = self.mass * (-self.g + az_des) if az_des is not None else 0  # Allow free fall if no desired acceleration
        print("u_f: ", u_f)

        return [u_f, tau_phi, tau_theta, tau_psi]
    

########################################################################
# UAV Simulation Class
########################################################################
class DroneSimulation:
    def __init__(self, mass, inertia, drag_coeffs, gravity):
        self.m = mass
        self.Ix, self.Iy, self.Iz = inertia
        self.J = np.diag([self.Ix, self.Iy, self.Iz])
        self.k_t, self.k_r = drag_coeffs
        self.g = gravity

    def rigid_body_dynamics(self, t, state, forces):
        """
        UAV rigid-body dynamics with external forces.
        State: [x, y, z, dx, dy, dz, phi, theta, psi, p, q, r]
        Control input: [lift_force, tau_phi, tau_theta, tau_psi]
        """
        x, y, z, dx, dy, dz, phi, theta, psi, p, q, r = state
        u_f, tau_phi, tau_theta, tau_psi = forces[:4]  # Control inputs
        external_force = forces[4:]  # External forces (step, linear, or noise)

        ddx = (1 / self.m) * ((np.cos(phi) * np.cos(theta) * np.sin(theta) * u_f) +
                              np.sin(phi) * np.sin(psi) * u_f - self.k_t * dx) + external_force[0] / self.m
        ddy = (1 / self.m) * ((np.cos(phi) * np.sin(theta) * np.sin(psi) -
                              np.cos(psi) * np.sin(phi)) * u_f - self.k_t * dy) + external_force[1] / self.m
        ddz = (1 / self.m) * (np.cos(phi) * np.cos(theta) * u_f - self.m * self.g - self.k_t * dz) + external_force[2] / self.m
        if z <= 0 and u_f < self.m * self.g:
            dz = 0
            ddz = 0
        dp = (1 / self.Ix) * (-self.k_r * p - q * r * (self.Iz - self.Iy) + tau_phi)
        dq = (1 / self.Iy) * (-self.k_r * q - r * p * (self.Ix - self.Iz) + tau_theta)
        dr = (1 / self.Iz) * (-self.k_r * r - p * q * (self.Iy - self.Ix) + tau_psi)
        dphi = p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r
        dtheta = np.cos(phi) * q - np.sin(phi) * r
        dpsi = (1 / np.cos(theta)) * (np.sin(phi) * q + np.cos(phi) * r)
        return [dx, dy, dz, ddx, ddy, ddz, dphi, dtheta, dpsi, dp, dq, dr]

    def normalize_euler_angles(self, phi, theta, psi):
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        psi = (psi + np.pi) % (2 * np.pi) - np.pi
        return phi, theta, psi

    def simulate(self, initial_state, forces, time_span, time_eval, external_forces, callback=None):
        """
        Simulate the drone dynamics with external forces.
        Forces include both control inputs and external disturbances.
        """
        integrator = RK4Integrator(self.rigid_body_dynamics, forces)
        times, states = integrator.integrate(time_eval, initial_state, callback)
        self.solution = type('Solution', (), {})()
        self.solution.y = states.T
        self.time_eval = times
        self.solution.y[6], self.solution.y[7], self.solution.y[8] = \
            self.normalize_euler_angles(self.solution.y[6], self.solution.y[7], self.solution.y[8])

    def data_results(self):
        solution = self.solution
        x, y, z = solution.y[0], solution.y[1], solution.y[2]
        dx, dy, dz = solution.y[3], solution.y[4], solution.y[5]
        phi, theta, psi = solution.y[6], solution.y[7], solution.y[8]
        p, q, r = solution.y[9], solution.y[10], solution.y[11]
        return x, y, z, dx, dy, dz, phi, theta, psi, p, q, r

    def plot_results(self):
        solution = self.solution
        x, y, z = solution.y[0], solution.y[1], solution.y[2]
        dx, dy, dz = solution.y[3], solution.y[4], solution.y[5]
        phi, theta, psi = solution.y[6], solution.y[7], solution.y[8]
        p, q, r = solution.y[9], solution.y[10], solution.y[11]

        fig, axs = plt.subplots(4, 1, figsize=(10, 10))
        axs[0].plot(self.time_eval, x, label='x')
        axs[0].plot(self.time_eval, y, label='y')
        axs[0].plot(self.time_eval, z, label='z')
        axs[0].set_title('Position over Time')
        axs[0].legend()

        axs[1].plot(self.time_eval, dx, label='dx')
        axs[1].plot(self.time_eval, dy, label='dy')
        axs[1].plot(self.time_eval, dz, label='dz')
        axs[1].set_title('Velocity over Time')
        axs[1].legend()

        axs[2].plot(self.time_eval, phi, label='phi')
        axs[2].plot(self.time_eval, theta, label='theta')
        axs[2].plot(self.time_eval, psi, label='psi')
        axs[2].set_title('Euler Angles over Time')
        axs[2].legend()

        axs[3].plot(self.time_eval, p, label='p (Roll rate)')
        axs[3].plot(self.time_eval, q, label='q (Pitch rate)')
        axs[3].plot(self.time_eval, r, label='r (Yaw rate)')
        axs[3].set_title('Angular Rates over Time')
        axs[3].legend()

        plt.tight_layout()
        plt.show()

    def animate_trajectory(self, animation_speed=1.0):
        """
        Animate the UAV trajectory.
        The animation speed is adjusted by the factor animation_speed.
        The interval is computed as base_interval / animation_speed.
        """
        solution = self.solution
        x, y, z = solution.y[0], solution.y[1], solution.y[2]
        phi, theta, psi = solution.y[6], solution.y[7], solution.y[8]

        # Base interval is 50 ms; adjust by animation_speed factor.
        interval = int(50 / animation_speed) if animation_speed > 0 else 50

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(0, 10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("UAV Trajectory Visualization")

        trajectory_line, = ax.plot([], [], [], 'b-', label="Trajectory")
        real_time_axes = []

        def draw_axes(center, R, length=1, alpha=0.8):
            colors = ['r', 'g', 'b']
            axes = []
            for i in range(3):
                start = center
                end = center + length * R[:, i]
                axes.append(Line3D([start[0], end[0]], 
                                   [start[1], end[1]], 
                                   [start[2], end[2]], 
                                   color=colors[i], alpha=alpha))
                ax.add_line(axes[-1])
            return axes

        initial_position = np.array([x[0], y[0], z[0]])
        initial_phi, initial_theta, initial_psi = phi[0], theta[0], psi[0]
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
        draw_axes(initial_position, R_initial, alpha=1.0)

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

        ani = FuncAnimation(fig, update, frames=len(self.time_eval), init_func=init, blit=False, interval=interval)
        plt.legend()
        plt.show()

    def apply_external_force(self, force_type="step", direction=None, magnitude=None, noise_level=0.0):
        """
        Apply external forces to the UAV.
        
        :param force_type: Type of external force ('step', 'linear', 'noise')
        :param direction: Direction of the force (for 'step' or 'linear' forces, should be a 3D vector)
        :param magnitude: Magnitude of the force (for 'step' or 'linear' forces, a scalar)
        :param noise_level: Amplitude of noise (for 'noise' force type)
        """
        if force_type == "step":
            if direction is None or magnitude is None:
                raise ValueError("Direction and magnitude must be specified for 'step' force.")
            return direction * magnitude
        elif force_type == "linear":
            if direction is None or magnitude is None:
                raise ValueError("Direction and magnitude must be specified for 'linear' force.")
            return direction * magnitude * np.linspace(0, 1, len(self.time_eval))
        elif force_type == "noise":
            noise = np.random.normal(0, noise_level, size=(3,))
            return noise
        else:
            raise ValueError("Invalid force type. Must be 'step', 'linear', or 'noise'.")
        
        
########################################################################
# CSV Exporter Class
########################################################################
class CSVExporter:
    """
    Exports the UAV state and control inputs at each time step to a CSV file.
    """
    def __init__(self, filename, headers=None):
        """
        Initializes the CSVExporter object with a filename and optional headers.
        If no headers are provided, default headers are used.
        """
        if headers is None:
            self.headers = ['time', 'x', 'y', 'z', 'dx', 'dy', 'dz',
                            'phi', 'theta', 'psi', 'p', 'q', 'r',
                            'lift_force', 'tau_phi', 'tau_theta', 'tau_psi']
        else:
            self.headers = headers
        self.filename = filename

    def export(self, time_eval, state_matrix, forces):
        """
        Exports the UAV state and control inputs at each time step to a CSV file.
        """
        n_steps = len(time_eval)
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            for i in range(n_steps):
                row = [
                    time_eval[i],
                    state_matrix[0][i],
                    state_matrix[1][i],
                    state_matrix[2][i],
                    state_matrix[3][i],
                    state_matrix[4][i],
                    state_matrix[5][i],
                    state_matrix[6][i],
                    state_matrix[7][i],
                    state_matrix[8][i],
                    state_matrix[9][i],
                    state_matrix[10][i],
                    state_matrix[11][i],
                    forces[0],
                    forces[1],
                    forces[2],
                    forces[3]
                ]
                writer.writerow(row)
                
                
########################################################################
# PID Callback Handler Class
########################################################################
class PIDCallbackHandler:
    """
    Encapsulates the callback function for the simulation.
    All callback operations (printing state, updating PID parameters, etc.)
    are implemented in this class.
    """
    def __init__(self, pid_controller, flight_mode="Fixed Point", trajectory_planner=None):
        self.pid_controller = pid_controller
        self.iteration_count = 0
        self.flight_mode = flight_mode  # "Fixed Point" or "Curve Tracking"
        self.trajectory_planner = trajectory_planner

    def callback(self, current_time, current_state, current_forces):
        self.iteration_count += 1
        # print("Iteration: {}, Time: {:.3f}, State: {}".format(self.iteration_count, current_time, current_state))
        # Example: update PID parameter Kp_x dynamically if needed
        # self.pid_controller.Kp_x = 1.0 + 0.0001 * self.iteration_count
        
        # If in Curve Tracking mode, update desired position based on trajectory
        if self.flight_mode == "Curve Tracking" and self.trajectory_planner is not None:
            new_target = self.trajectory_planner.get_target_position(current_time)
            self.pid_controller.desired_position = new_target
        new_forces = self.pid_controller.update(current_time, current_state)
        return new_forces
    
    
# ----------------- GUI Code -----------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("UAV Simulation Parameters")
        # Create a Notebook with three tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)
        
        # Drone Parameters Tab
        self.drone_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.drone_frame, text="Drone Parameters")
        self.drone_params = {
            "mass": config.get("DroneSimulation", "mass"),
            "inertia_x": config.get("DroneSimulation", "inertia_x"),
            "inertia_y": config.get("DroneSimulation", "inertia_y"),
            "inertia_z": config.get("DroneSimulation", "inertia_z"),
            "drag_coeff_linear": config.get("DroneSimulation", "drag_coeff_linear"),
            "drag_coeff_angular": config.get("DroneSimulation", "drag_coeff_angular"),
            "gravity": config.get("DroneSimulation", "gravity")
        }
        self.drone_entries = {}
        self.create_entries(self.drone_frame, self.drone_params, row=0)
        
        # Simulation Parameters Tab
        self.sim_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sim_frame, text="Simulation Parameters")
        self.sim_params = {
            "initial_state_x": config.get("Simulation", "initial_state_x"),
            "initial_state_y": config.get("Simulation", "initial_state_y"),
            "initial_state_z": config.get("Simulation", "initial_state_z"),
            "initial_state_dx": config.get("Simulation", "initial_state_dx"),
            "initial_state_dy": config.get("Simulation", "initial_state_dy"),
            "initial_state_dz": config.get("Simulation", "initial_state_dz"),
            "initial_state_phi": config.get("Simulation", "initial_state_phi"),
            "initial_state_theta": config.get("Simulation", "initial_state_theta"),
            "initial_state_psi": config.get("Simulation", "initial_state_psi"),
            "initial_state_p": config.get("Simulation", "initial_state_p"),
            "initial_state_q": config.get("Simulation", "initial_state_q"),
            "initial_state_r": config.get("Simulation", "initial_state_r"),
            "target_position_x": config.get("Simulation", "target_position_x"),
            "target_position_y": config.get("Simulation", "target_position_y"),
            "target_position_z": config.get("Simulation", "target_position_z"),
            "target_velocity_x": config.get("Simulation", "target_velocity_x"),
            "target_velocity_y": config.get("Simulation", "target_velocity_y"),
            "target_velocity_z": config.get("Simulation", "target_velocity_z"),
            "time_span_start": config.get("Simulation", "time_span_start"),
            "time_span_end": config.get("Simulation", "time_span_end"),
            "time_eval_points": config.get("Simulation", "time_eval_points"),
            "forces_u_f": config.get("Simulation", "forces_u_f"),
            "forces_tau_phi": config.get("Simulation", "forces_tau_phi"),
            "forces_tau_theta": config.get("Simulation", "forces_tau_theta"),
            "forces_tau_psi": config.get("Simulation", "forces_tau_psi"),
            "animation_speed": config.get("Simulation", "animation_speed")  # New parameter for animation speed
        }
        self.sim_entries = {}
        self.create_entries(self.sim_frame, self.sim_params, row=0)
        
        # PID Parameters Tab
        self.pid_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.pid_frame, text="PID Parameters")
        self.pid_params = {
            # Outer loop (Position control)
            "kp_x": config.get("PIDController", "kp_x"),
            "ki_x": config.get("PIDController", "ki_x"),
            "kd_x": config.get("PIDController", "kd_x"),
            "kp_y": config.get("PIDController", "kp_y"),
            "ki_y": config.get("PIDController", "ki_y"),
            "kd_y": config.get("PIDController", "kd_y"),
            "kp_z": config.get("PIDController", "kp_z"),
            "ki_z": config.get("PIDController", "ki_z"),
            "kd_z": config.get("PIDController", "kd_z"),
            # Second loop (Velocity control)
            "kp_vx": config.get("PIDController", "kp_vx"),
            "ki_vx": config.get("PIDController", "ki_vx"),
            "kd_vx": config.get("PIDController", "kd_vx"),
            "kp_vy": config.get("PIDController", "kp_vy"),
            "ki_vy": config.get("PIDController", "ki_vy"),
            "kd_vy": config.get("PIDController", "kd_vy"),
            "kp_vz": config.get("PIDController", "kp_vz"),
            "ki_vz": config.get("PIDController", "ki_vz"),
            "kd_vz": config.get("PIDController", "kd_vz"),
            # Middle loop (Attitude control)
            "att_kp_phi": config.get("PIDController", "att_kp_phi"),
            "att_ki_phi": config.get("PIDController", "att_ki_phi"),
            "att_kd_phi": config.get("PIDController", "att_kd_phi"),
            "att_kp_theta": config.get("PIDController", "att_kp_theta"),
            "att_ki_theta": config.get("PIDController", "att_ki_theta"),
            "att_kd_theta": config.get("PIDController", "att_kd_theta"),
            "att_kp_psi": config.get("PIDController", "att_kp_psi"),
            "att_ki_psi": config.get("PIDController", "att_ki_psi"),
            "att_kd_psi": config.get("PIDController", "att_kd_psi"),
            # Inner loop (Angular rate control)
            "rate_kp_phi": config.get("PIDController", "rate_kp_phi"),
            "rate_ki_phi": config.get("PIDController", "rate_ki_phi"),
            "rate_kd_phi": config.get("PIDController", "rate_kd_phi"),
            "rate_kp_theta": config.get("PIDController", "rate_kp_theta"),
            "rate_ki_theta": config.get("PIDController", "rate_ki_theta"),
            "rate_kd_theta": config.get("PIDController", "rate_kd_theta"),
            "rate_kp_psi": config.get("PIDController", "rate_kp_psi"),
            "rate_ki_psi": config.get("PIDController", "rate_ki_psi"),
            "rate_kd_psi": config.get("PIDController", "rate_kd_psi"),
            "pid_dt": config.get("PIDController", "pid_dt"),
            # Target attitude (if desired_phi and desired_theta are 0, then computed from position control)
            "desired_phi": config.get("PIDController", "desired_phi"),
            "desired_theta": config.get("PIDController", "desired_theta"),
            "desired_psi": config.get("PIDController", "desired_psi")
        }
        self.pid_entries = {}
        self.create_entries(self.pid_frame, self.pid_params, row=0)

        # Run Simulation Button
        self.run_button = ttk.Button(root, text="Run Simulation", command=self.run_simulation)
        self.run_button.pack(pady=10)
        # Quit Button
        self.quit_button = ttk.Button(root, text="Quit", command=root.quit)
        self.quit_button.pack(pady=5)

    def create_entries(self, parent, param_dict, row=0):
        """Create a label and a Spinbox with increment 0.1 for each parameter."""
        for key, val in param_dict.items():
            lbl = ttk.Label(parent, text=key)
            lbl.grid(row=row, column=0, sticky="W", padx=5, pady=2)
            spin = tk.Spinbox(parent, from_=-10000, to=10000, increment=0.1, width=10)
            spin.delete(0, "end")
            spin.insert(0, str(val))
            spin.grid(row=row, column=1, padx=5, pady=2)
            param_dict[key] = spin
            row += 1

    def run_simulation(self):
        try:
            # Read Drone Parameters
            drone_params = {
                "mass": float(self.drone_params["mass"].get()),
                "inertia_x": float(self.drone_params["inertia_x"].get()),
                "inertia_y": float(self.drone_params["inertia_y"].get()),
                "inertia_z": float(self.drone_params["inertia_z"].get()),
                "drag_coeff_linear": float(self.drone_params["drag_coeff_linear"].get()),
                "drag_coeff_angular": float(self.drone_params["drag_coeff_angular"].get()),
                "gravity": float(self.drone_params["gravity"].get())
            }
            # Read Simulation Parameters
            sim_params = {
                "initial_state_x": float(self.sim_params["initial_state_x"].get()),
                "initial_state_y": float(self.sim_params["initial_state_y"].get()),
                "initial_state_z": float(self.sim_params["initial_state_z"].get()),
                "initial_state_dx": float(self.sim_params["initial_state_dx"].get()),
                "initial_state_dy": float(self.sim_params["initial_state_dy"].get()),
                "initial_state_dz": float(self.sim_params["initial_state_dz"].get()),
                "initial_state_phi": float(self.sim_params["initial_state_phi"].get()),
                "initial_state_theta": float(self.sim_params["initial_state_theta"].get()),
                "initial_state_psi": float(self.sim_params["initial_state_psi"].get()),
                "initial_state_p": float(self.sim_params["initial_state_p"].get()),
                "initial_state_q": float(self.sim_params["initial_state_q"].get()),
                "initial_state_r": float(self.sim_params["initial_state_r"].get()),
                "target_position_x": float(self.sim_params["target_position_x"].get()),
                "target_position_y": float(self.sim_params["target_position_y"].get()),
                "target_position_z": float(self.sim_params["target_position_z"].get()),
                "target_velocity_x": float(self.sim_params["target_velocity_x"].get()),
                "target_velocity_y": float(self.sim_params["target_velocity_y"].get()),
                "target_velocity_z": float(self.sim_params["target_velocity_z"].get()),
                "time_span_start": float(self.sim_params["time_span_start"].get()),
                "time_span_end": float(self.sim_params["time_span_end"].get()),
                "time_eval_points": int(self.sim_params["time_eval_points"].get()),
                "forces_u_f": float(self.sim_params["forces_u_f"].get()),
                "forces_tau_phi": float(self.sim_params["forces_tau_phi"].get()),
                "forces_tau_theta": float(self.sim_params["forces_tau_theta"].get()),
                "forces_tau_psi": float(self.sim_params["forces_tau_psi"].get()),
                "animation_speed": float(self.sim_params["animation_speed"].get())
            }
            # Read PID Parameters
            pid_params = {
                "kp_x": float(self.pid_params["kp_x"].get()),
                "ki_x": float(self.pid_params["ki_x"].get()),
                "kd_x": float(self.pid_params["kd_x"].get()),
                "kp_y": float(self.pid_params["kp_y"].get()),
                "ki_y": float(self.pid_params["ki_y"].get()),
                "kd_y": float(self.pid_params["kd_y"].get()),
                "kp_z": float(self.pid_params["kp_z"].get()),
                "ki_z": float(self.pid_params["ki_z"].get()),
                "kd_z": float(self.pid_params["kd_z"].get()),
                "kp_vx": float(self.pid_params["kp_vx"].get()),
                "ki_vx": float(self.pid_params["ki_vx"].get()),
                "kd_vx": float(self.pid_params["kd_vx"].get()),
                "kp_vy": float(self.pid_params["kp_vy"].get()),
                "ki_vy": float(self.pid_params["ki_vy"].get()),
                "kd_vy": float(self.pid_params["kd_vy"].get()),
                "kp_vz": float(self.pid_params["kp_vz"].get()),
                "ki_vz": float(self.pid_params["ki_vz"].get()),
                "kd_vz": float(self.pid_params["kd_vz"].get()),
                "att_kp_phi": float(self.pid_params["att_kp_phi"].get()),
                "att_ki_phi": float(self.pid_params["att_ki_phi"].get()),
                "att_kd_phi": float(self.pid_params["att_kd_phi"].get()),
                "att_kp_theta": float(self.pid_params["att_kp_theta"].get()),
                "att_ki_theta": float(self.pid_params["att_ki_theta"].get()),
                "att_kd_theta": float(self.pid_params["att_kd_theta"].get()),
                "att_kp_psi": float(self.pid_params["att_kp_psi"].get()),
                "att_ki_psi": float(self.pid_params["att_ki_psi"].get()),
                "att_kd_psi": float(self.pid_params["att_kd_psi"].get()),
                "rate_kp_phi": float(self.pid_params["rate_kp_phi"].get()),
                "rate_ki_phi": float(self.pid_params["rate_ki_phi"].get()),
                "rate_kd_phi": float(self.pid_params["rate_kd_phi"].get()),
                "rate_kp_theta": float(self.pid_params["rate_kp_theta"].get()),
                "rate_ki_theta": float(self.pid_params["rate_ki_theta"].get()),
                "rate_kd_theta": float(self.pid_params["rate_kd_theta"].get()),
                "rate_kp_psi": float(self.pid_params["rate_kp_psi"].get()),
                "rate_ki_psi": float(self.pid_params["rate_ki_psi"].get()),
                "rate_kd_psi": float(self.pid_params["rate_kd_psi"].get()),
                "pid_dt": float(self.pid_params["pid_dt"].get()),
                "desired_phi": float(self.pid_params["desired_phi"].get()),
                "desired_theta": float(self.pid_params["desired_theta"].get()),
                "desired_psi": float(self.pid_params["desired_psi"].get())
            }
            initial_state = [
                sim_params["initial_state_x"],
                sim_params["initial_state_y"],
                sim_params["initial_state_z"],
                sim_params["initial_state_dx"],
                sim_params["initial_state_dy"],
                sim_params["initial_state_dz"],
                sim_params["initial_state_phi"],
                sim_params["initial_state_theta"],
                sim_params["initial_state_psi"],
                sim_params["initial_state_p"],
                sim_params["initial_state_q"],
                sim_params["initial_state_r"]
            ]
            target_position = (
                sim_params["target_position_x"],
                sim_params["target_position_y"],
                sim_params["target_position_z"]
            )
            target_velocity = (
                sim_params["target_velocity_x"],
                sim_params["target_velocity_y"],
                sim_params["target_velocity_z"]
            )
            forces = [
                sim_params["forces_u_f"],
                sim_params["forces_tau_phi"],
                sim_params["forces_tau_theta"],
                sim_params["forces_tau_psi"]
            ]
            time_eval = np.linspace(sim_params["time_span_start"], sim_params["time_span_end"], sim_params["time_eval_points"])
            
            global pid_controller, iteration_count
            iteration_count = 0
            pid_controller = DualLoopPIDController(
                mass=drone_params["mass"],
                gravity=drone_params["gravity"],
                desired_position=target_position,
                desired_velocity=target_velocity,
                desired_attitude=(pid_params["desired_phi"], pid_params["desired_theta"], pid_params["desired_psi"]),
                dt=pid_params["pid_dt"],
                kp_x=pid_params["kp_x"], ki_x=pid_params["ki_x"], kd_x=pid_params["kd_x"],
                kp_y=pid_params["kp_y"], ki_y=pid_params["ki_y"], kd_y=pid_params["kd_y"],
                kp_z=pid_params["kp_z"], ki_z=pid_params["ki_z"], kd_z=pid_params["kd_z"],
                kp_vx=pid_params["kp_vx"], ki_vx=pid_params["ki_vx"], kd_vx=pid_params["kd_vx"],
                kp_vy=pid_params["kp_vy"], ki_vy=pid_params["ki_vy"], kd_vy=pid_params["kd_vy"],
                kp_vz=pid_params["kp_vz"], ki_vz=pid_params["ki_vz"], kd_vz=pid_params["kd_vz"],
                att_kp_phi=pid_params["att_kp_phi"], att_ki_phi=pid_params["att_ki_phi"], att_kd_phi=pid_params["att_kd_phi"],
                att_kp_theta=pid_params["att_kp_theta"], att_ki_theta=pid_params["att_ki_theta"], att_kd_theta=pid_params["att_kd_theta"],
                att_kp_psi=pid_params["att_kp_psi"], att_ki_psi=pid_params["att_ki_psi"], att_kd_psi=pid_params["att_kd_psi"],
                rate_kp_phi=pid_params["rate_kp_phi"], rate_ki_phi=pid_params["rate_ki_phi"], rate_kd_phi=pid_params["rate_kd_phi"],
                rate_kp_theta=pid_params["rate_kp_theta"], rate_ki_theta=pid_params["rate_ki_theta"], rate_kd_theta=pid_params["rate_kd_theta"],
                rate_kp_psi=pid_params["rate_kp_psi"], rate_ki_psi=pid_params["rate_ki_psi"], rate_kd_psi=pid_params["rate_kd_psi"]
            )
            
            # Create PID callback handler instance
            callback_handler = PIDCallbackHandler(pid_controller)
            
            drone = DroneSimulation(
                mass=drone_params["mass"],
                inertia=(drone_params["inertia_x"], drone_params["inertia_y"], drone_params["inertia_z"]),
                drag_coeffs=(drone_params["drag_coeff_linear"], drone_params["drag_coeff_angular"]),
                gravity=drone_params["gravity"]
            )
            drone.simulate(initial_state, 
                           forces, 
                           (sim_params["time_span_start"], sim_params["time_span_end"]),
                           time_eval, 
                           external_forces=drone.apply_external_force(force_type='noise', direction=1 ,magnitude=1, noise_level=0), 
                           callback=callback_handler.callback)
            
            csv_exporter = CSVExporter("simulation_results.csv")
            csv_exporter.export(time_eval, drone.solution.y, forces)
            
            drone.plot_results()
            # Pass animation speed from simulation parameters to animate_trajectory
            animation_speed = sim_params["animation_speed"]
            drone.animate_trajectory(animation_speed=animation_speed)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
            
def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()