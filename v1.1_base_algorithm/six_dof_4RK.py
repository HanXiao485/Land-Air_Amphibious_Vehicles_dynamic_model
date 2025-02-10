import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D
import configparser
import csv
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Read configuration file
config = configparser.ConfigParser()
config.read('E:\\Land-Air_Amphibious_Vehicles_dynamic_model\\v1.1\\config.ini')

########################################################################
# PID Controller Class with Three Loops (Position, Attitude, Rate)
########################################################################
class DualLoopPIDController:
    """
    Three-loop PID controller:
    - Outer loop (Position control): calculates desired acceleration based on position error,
      and then computes the desired roll and pitch angles using small-angle approximation.
    - Middle loop (Attitude control): calculates desired angular rates based on attitude error
      using full PID control (with proportional, integral and derivative terms).
    - Inner loop (Rate control): calculates control moments using PID control on angular rate error.
    
    The total lift force is computed from the vertical acceleration:
         u_f = mass * (g + a_z_des)
    """
    def __init__(self, mass, gravity, desired_position, desired_attitude, dt,
                 # Outer loop PID parameters (Position control)
                 kp_x=1.0, ki_x=0.0, kd_x=0.5,
                 kp_y=1.0, ki_y=0.0, kd_y=0.5,
                 kp_z=2.0, ki_z=0.0, kd_z=1.0,
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
        self.desired_attitude = desired_attitude      # Target attitude: (phi_des, theta_des, psi_des)
        self.dt = dt

        # Outer loop PID parameters (Position control)
        self.Kp_x = kp_x; self.Ki_x = ki_x; self.Kd_x = kd_x
        self.Kp_y = kp_y; self.Ki_y = ki_y; self.Kd_y = kd_y
        self.Kp_z = kp_z; self.Ki_z = ki_z; self.Kd_z = kd_z

        # Middle loop PID parameters (Attitude control)
        self.att_kp_phi = att_kp_phi; self.att_ki_phi = att_ki_phi; self.att_kd_phi = att_kd_phi
        self.att_kp_theta = att_kp_theta; self.att_ki_theta = att_ki_theta; self.att_kd_theta = att_kd_theta
        self.att_kp_psi = att_kp_psi; self.att_ki_psi = att_ki_psi; self.att_kd_psi = att_kd_psi

        # Initialize middle loop integration and last error (Attitude control)
        self.int_phi_att = 0.0; self.last_error_phi_att = 0.0
        self.int_theta_att = 0.0; self.last_error_theta_att = 0.0
        self.int_psi_att = 0.0; self.last_error_psi_att = 0.0

        # Inner loop PID parameters (Angular rate control)
        self.rate_kp_phi = rate_kp_phi; self.rate_ki_phi = rate_ki_phi; self.rate_kd_phi = rate_kd_phi
        self.rate_kp_theta = rate_kp_theta; self.rate_ki_theta = rate_ki_theta; self.rate_kd_theta = rate_kd_theta
        self.rate_kp_psi = rate_kp_psi; self.rate_ki_psi = rate_ki_psi; self.rate_kd_psi = rate_kd_psi

        # Initialize outer loop integration and last error (Position control)
        self.int_x = 0.0; self.last_error_x = 0.0
        self.int_y = 0.0; self.last_error_y = 0.0
        self.int_z = 0.0; self.last_error_z = 0.0

        # Initialize inner loop integration and last error (Angular rate control)
        self.int_p = 0.0; self.last_error_p = 0.0
        self.int_q = 0.0; self.last_error_q = 0.0
        self.int_r = 0.0; self.last_error_r = 0.0

        self.last_time = None

    def update(self, current_time, state):
        """
        Compute new control input based on current time and state.
        State: [x, y, z, dx, dy, dz, phi, theta, psi, p, q, r]
        Output: Control inputs: [lift_force, tau_phi, tau_theta, tau_psi]
        """
        if self.last_time is None:
            dt = self.dt
        else:
            dt = current_time - self.last_time
        self.last_time = current_time

        # Extract state variables
        x, y, z, dx, dy, dz, phi, theta, psi, p, q, r = state
        x_des, y_des, z_des = self.desired_position

        # Outer loop: Position control
        error_x = x_des - x
        error_y = y_des - y
        error_z = z_des - z
        print("error_x: ", error_x)
    

        self.int_x += error_x * dt
        self.int_y += error_y * dt
        self.int_z += error_z * dt

        d_error_x = (error_x - self.last_error_x) / dt
        d_error_y = (error_y - self.last_error_y) / dt
        d_error_z = (error_z - self.last_error_z) / dt

        self.last_error_x = error_x
        self.last_error_y = error_y
        self.last_error_z = error_z

        # Compute desired acceleration
        ax_des = self.Kp_x * error_x + self.Ki_x * self.int_x + self.Kd_x * d_error_x
        ay_des = self.Kp_y * error_y + self.Ki_y * self.int_y + self.Kd_y * d_error_y
        az_des = self.Kp_z * error_z + self.Ki_z * self.int_z + self.Kd_z * d_error_z

        # Compute computed desired attitude from position control using small-angle approximation
        phi_des_pos = (1.0 / self.g) * ay_des
        theta_des_pos = (1.0 / self.g) * ax_des

        # If desired attitude for roll and pitch is set to 0, use computed values for horizontal motion.
        # Otherwise, use the provided desired attitude.
        if self.desired_attitude[0] == 0:
            phi_des = phi_des_pos
        else:
            phi_des = self.desired_attitude[0]
        if self.desired_attitude[1] == 0:
            theta_des = theta_des_pos
        else:
            theta_des = self.desired_attitude[1]
        # Yaw is always taken from desired attitude.
        psi_des = self.desired_attitude[2]

        # Middle loop: Attitude control (calculate desired angular rates with full PID)
        error_phi = phi_des - phi
        error_theta = theta_des - theta
        error_psi = psi_des - psi
        
        print("Error theta: ", error_theta)

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

        # Inner loop: Rate control
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

        # Compute total lift force
        u_f = self.mass * (self.g + az_des)

        return [u_f, tau_phi, tau_theta, tau_psi]

########################################################################
# Global PID Controller instance (to be initialized in main)
########################################################################
pid_controller = None
iteration_count = 0

def pid_callback(current_time, current_state, current_forces):
    """
    Callback function called after each integration step:
      1. Print current time and UAV state.
      2. Update PID parameters if needed.
      3. Return updated control input.
    """
    global pid_controller, iteration_count
    iteration_count += 1
    # print("Iteration: {}, Time: {:.3f}, State: {}".format(iteration_count, current_time, current_state))
    
    # Example: update PID parameters if needed (e.g., adjust position control Kp_x)
    pid_controller.Kp_x = 1.0 + 0.0001 * iteration_count
    
    new_forces = pid_controller.update(current_time, current_state)
    return new_forces

########################################################################
# RK4 Integrator Class (with callback)
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
            # Prevent UAV from penetrating the ground: if z < 0, set z and dz to 0
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
# CSV Exporter Class
########################################################################
class CSVExporter:
    """
    Exports the UAV state and control inputs at each time step to a CSV file.
    """
    def __init__(self, filename, headers=None):
        if headers is None:
            self.headers = ['time', 'x', 'y', 'z', 'dx', 'dy', 'dz',
                            'phi', 'theta', 'psi', 'p', 'q', 'r',
                            'lift_force', 'tau_phi', 'tau_theta', 'tau_psi']
        else:
            self.headers = headers
        self.filename = filename

    def export(self, time_eval, state_matrix, forces):
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
        UAV rigid-body dynamics.
        State: [x, y, z, dx, dy, dz, phi, theta, psi, p, q, r]
        Control input: [lift_force, tau_phi, tau_theta, tau_psi]
        """
        x, y, z, dx, dy, dz, phi, theta, psi, p, q, r = state
        u_f, tau_phi, tau_theta, tau_psi = forces

        ddx = (1 / self.m) * ((np.cos(phi) * np.cos(theta) * np.sin(theta) * u_f) +
                              np.sin(phi) * np.sin(psi) * u_f - self.k_t * dx)
        ddy = (1 / self.m) * ((np.cos(phi) * np.sin(theta) * np.sin(psi) -
                              np.cos(psi) * np.sin(phi)) * u_f - self.k_t * dy)
        ddz = (1 / self.m) * (np.cos(phi) * np.cos(theta) * u_f - self.m * self.g - self.k_t * dz)
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

    def simulate(self, initial_state, forces, time_span, time_eval, callback=None):
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

    def animate_trajectory(self):
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

        ani = FuncAnimation(fig, update, frames=len(self.time_eval), init_func=init, blit=False, interval=50)
        plt.legend()
        plt.show()

########################################################################
# Main Function
########################################################################
def main():
    # Read UAV and simulation parameters from config file
    mass = config.getfloat('DroneSimulation', 'mass')
    inertia = (
        config.getfloat('DroneSimulation', 'inertia_x'),
        config.getfloat('DroneSimulation', 'inertia_y'),
        config.getfloat('DroneSimulation', 'inertia_z')
    )
    drag_coeffs = (
        config.getfloat('DroneSimulation', 'drag_coeff_linear'),
        config.getfloat('DroneSimulation', 'drag_coeff_angular')
    )
    gravity = config.getfloat('DroneSimulation', 'gravity')

    # Read initial state from [Simulation]
    initial_state = [
        config.getfloat('Simulation', 'initial_state_x'),
        config.getfloat('Simulation', 'initial_state_y'),
        config.getfloat('Simulation', 'initial_state_z'),
        config.getfloat('Simulation', 'initial_state_dx'),
        config.getfloat('Simulation', 'initial_state_dy'),
        config.getfloat('Simulation', 'initial_state_dz'),
        config.getfloat('Simulation', 'initial_state_phi'),
        config.getfloat('Simulation', 'initial_state_theta'),
        config.getfloat('Simulation', 'initial_state_psi'),
        config.getfloat('Simulation', 'initial_state_p'),
        config.getfloat('Simulation', 'initial_state_q'),
        config.getfloat('Simulation', 'initial_state_r')
    ]

    # Read target position from [Simulation]
    target_position = (
        config.getfloat('Simulation', 'target_position_x'),
        config.getfloat('Simulation', 'target_position_y'),
        config.getfloat('Simulation', 'target_position_z')
    )

    # Read PID parameters from [PIDController], including target attitude
    pid_params = {
        'kp_x': config.getfloat('PIDController', 'kp_x'),
        'ki_x': config.getfloat('PIDController', 'ki_x'),
        'kd_x': config.getfloat('PIDController', 'kd_x'),
        'kp_y': config.getfloat('PIDController', 'kp_y'),
        'ki_y': config.getfloat('PIDController', 'ki_y'),
        'kd_y': config.getfloat('PIDController', 'kd_y'),
        'kp_z': config.getfloat('PIDController', 'kp_z'),
        'ki_z': config.getfloat('PIDController', 'ki_z'),
        'kd_z': config.getfloat('PIDController', 'kd_z'),
        'att_kp_phi': config.getfloat('PIDController', 'att_kp_phi'),
        'att_ki_phi': config.getfloat('PIDController', 'att_ki_phi'),
        'att_kd_phi': config.getfloat('PIDController', 'att_kd_phi'),
        'att_kp_theta': config.getfloat('PIDController', 'att_kp_theta'),
        'att_ki_theta': config.getfloat('PIDController', 'att_ki_theta'),
        'att_kd_theta': config.getfloat('PIDController', 'att_kd_theta'),
        'att_kp_psi': config.getfloat('PIDController', 'att_kp_psi'),
        'att_ki_psi': config.getfloat('PIDController', 'att_ki_psi'),
        'att_kd_psi': config.getfloat('PIDController', 'att_kd_psi'),
        'rate_kp_phi': config.getfloat('PIDController', 'rate_kp_phi'),
        'rate_ki_phi': config.getfloat('PIDController', 'rate_ki_phi'),
        'rate_kd_phi': config.getfloat('PIDController', 'rate_kd_phi'),
        'rate_kp_theta': config.getfloat('PIDController', 'rate_kp_theta'),
        'rate_ki_theta': config.getfloat('PIDController', 'rate_ki_theta'),
        'rate_kd_theta': config.getfloat('PIDController', 'rate_kd_theta'),
        'rate_kp_psi': config.getfloat('PIDController', 'rate_kp_psi'),
        'rate_ki_psi': config.getfloat('PIDController', 'rate_ki_psi'),
        'rate_kd_psi': config.getfloat('PIDController', 'rate_kd_psi'),
        'pid_dt': config.getfloat('PIDController', 'pid_dt'),
        # Target attitude; if not set (None or 0 for phi/theta), the computed value from position control is used.
        'desired_phi': config.getfloat('PIDController', 'desired_phi') if config.has_option('PIDController', 'desired_phi') else None,
        'desired_theta': config.getfloat('PIDController', 'desired_theta') if config.has_option('PIDController', 'desired_theta') else None,
        'desired_psi': config.getfloat('PIDController', 'desired_psi')
    }

    # Users can override parameters here (example):
    # initial_state[0] = 1.0
    # target_position = (0.0, 0.0, 10.0)
    # pid_params['kp_x'] = 2.0

    forces = [
        config.getfloat('Simulation', 'forces_u_f'),
        config.getfloat('Simulation', 'forces_tau_phi'),
        config.getfloat('Simulation', 'forces_tau_theta'),
        config.getfloat('Simulation', 'forces_tau_psi')
    ]

    time_span = (
        config.getfloat('Simulation', 'time_span_start'),
        config.getfloat('Simulation', 'time_span_end')
    )
    time_eval = np.linspace(time_span[0], time_span[1], config.getint('Simulation', 'time_eval_points'))

    global pid_controller, iteration_count
    iteration_count = 0
    pid_controller = DualLoopPIDController(
        mass, gravity, target_position,
        desired_attitude=(pid_params['desired_phi'], pid_params['desired_theta'], pid_params['desired_psi']),
        dt=pid_params['pid_dt'],
        kp_x=pid_params['kp_x'], ki_x=pid_params['ki_x'], kd_x=pid_params['kd_x'],
        kp_y=pid_params['kp_y'], ki_y=pid_params['ki_y'], kd_y=pid_params['kd_y'],
        kp_z=pid_params['kp_z'], ki_z=pid_params['ki_z'], kd_z=pid_params['kd_z'],
        att_kp_phi=pid_params['att_kp_phi'], att_ki_phi=pid_params['att_ki_phi'], att_kd_phi=pid_params['att_kd_phi'],
        att_kp_theta=pid_params['att_kp_theta'], att_ki_theta=pid_params['att_ki_theta'], att_kd_theta=pid_params['att_kd_theta'],
        att_kp_psi=pid_params['att_kp_psi'], att_ki_psi=pid_params['att_ki_psi'], att_kd_psi=pid_params['att_kd_psi'],
        rate_kp_phi=pid_params['rate_kp_phi'], rate_ki_phi=pid_params['rate_ki_phi'], rate_kd_phi=pid_params['rate_kd_phi'],
        rate_kp_theta=pid_params['rate_kp_theta'], rate_ki_theta=pid_params['rate_ki_theta'], rate_kd_theta=pid_params['rate_kd_theta'],
        rate_kp_psi=pid_params['rate_kp_psi'], rate_ki_psi=pid_params['rate_ki_psi'], rate_kd_psi=pid_params['rate_kd_psi']
    )

    drone = DroneSimulation(mass, inertia, drag_coeffs, gravity)
    drone.simulate(initial_state, forces, time_span, time_eval, callback=pid_callback)

    csv_exporter = CSVExporter("simulation_results.csv")
    csv_exporter.export(time_eval, drone.solution.y, forces)

    drone.plot_results()
    drone.animate_trajectory()


if __name__ == "__main__":
    main()
