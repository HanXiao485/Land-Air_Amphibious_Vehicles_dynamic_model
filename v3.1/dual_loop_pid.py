import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D
import configparser
import csv
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

from curve import Curve

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
    def __init__(self, mass, gravity, desired_position, desired_velocity:Curve, desired_attitude, dt,
                 # Outer loop PID parameters (Position control)
                 kp_x=0.0, ki_x=0.0, kd_x=0.0,
                 kp_y=0.0, ki_y=0.0, kd_y=0.0,
                 kp_z=0.0, ki_z=0.0, kd_z=0.0,
                 # Second loop PID parameters (Velocity control)
                 kp_vx=0.0, ki_vx=0.0, kd_vx=0.0,
                 kp_vy=0.0, ki_vy=0.0, kd_vy=0.0,
                 kp_vz=0.0, ki_vz=0.0, kd_vz=0.0,
                 # Middle loop PID parameters (Attitude control)
                 att_kp_phi=0.0, att_ki_phi=0.0, att_kd_phi=0.0,
                 att_kp_theta=5.0, att_ki_theta=0.1, att_kd_theta=0.0,
                 att_kp_psi=0.0, att_ki_psi=0.0, att_kd_psi=0.0,
                 # Inner loop PID parameters (Angular rate control)
                 rate_kp_phi=0.0, rate_ki_phi=0.0, rate_kd_phi=0.0,
                 rate_kp_theta=0.0, rate_ki_theta=0.0, rate_kd_theta=0.0,
                 rate_kp_psi=0.0, rate_ki_psi=0.0, rate_kd_psi=0.0):
        self.mass = mass
        self.g = gravity
        self.desired_position = desired_position      # Target position: (x_des, y_des, z_des)
        self.desired_velocity = desired_velocity      # Target velocity: (vx_des, vy_des, vz_des)
        self.desired_attitude = desired_attitude      # Target attitude: (phi_des, theta_des, psi_des)
        self.dt = dt
        
        # initialize PID controller parameters
        self.pid_params = np.zeros(36)

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
        
        self.z_des_list = []
        
        
    def set_pid_params(self, params):
        """将输入参数映射到合理范围"""
        # # 使用sigmoid缩放比例项到(0, 10)
        # self.pid_params[:18] = torch.sigmoid(torch.tensor(params[:18]))
        # # 使用tanh缩放微分项到(-5,5)
        # self.pid_params[18:] = torch.tanh(torch.tensor(params[18:]))
        self.pid_params = params
        
        # self.Kp_x = self.pid_params[0]; self.Ki_x = self.pid_params[1]; self.Kd_x = self.pid_params[2]
        # self.Kp_y = self.pid_params[3]; self.Ki_y = self.pid_params[4]; self.Kd_y = self.pid_params[5]
        # self.Kp_z = self.pid_params[6]; self.Ki_z = self.pid_params[7]; self.Kd_z = self.pid_params[8]
        # self.Kp_vx = self.pid_params[9]; self.Ki_vx = self.pid_params[10]; self.Kd_vx = self.pid_params[11]
        # self.Kp_vy = self.pid_params[12]; self.Ki_vy = self.pid_params[13]; self.Kd_vy = self.pid_params[14]
        # self.Kp_vz = self.pid_params[15]; self.Ki_vz = self.pid_params[16]; self.Kd_vz = self.pid_params[17]
        # self.rate_kp_phi = self.pid_params[18]; self.rate_ki_phi = self.pid_params[19]; self.rate_kd_phi = self.pid_params[20]
        # self.rate_kp_theta = self.pid_params[21]; self.rate_ki_theta = self.pid_params[22]; self.rate_kd_theta = self.pid_params[23]
        # self.rate_kp_psi = self.pid_params[24]; self.rate_ki_psi = self.pid_params[25]; self.rate_kd_psi = self.pid_params[26]
        # self.att_kp_phi = self.pid_params[27]; self.att_ki_phi = self.pid_params[28]; self.att_kd_phi = self.pid_params[29]
        # self.att_kp_theta = self.pid_params[30]; self.att_ki_theta = self.pid_params[31]; self.att_kd_theta = self.pid_params[32]
        # self.att_kp_psi = self.pid_params[33]; self.att_ki_psi = self.pid_params[34]; self.att_kd_psi = self.pid_params[35]

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
        
        # Extract PID parameters
        # (self.Kp_x, self.Ki_x, self.Kd_x, self.Kp_y, self.Ki_y, self.Kd_y, self.Kp_z, self.Ki_z, self.Kd_z, 
        #  self.Kp_vx, self.Ki_vx, self.Kd_vx, self.Kp_vy, self.Ki_vy, self.Kd_vy, self.Kp_vz, self.Ki_vz, self.Kd_vz,
        #  self.rate_kp_phi, self.rate_ki_phi, self.rate_kd_phi, self.rate_kp_theta, self.rate_ki_theta, self.rate_kd_theta, self.rate_kp_psi, self.rate_ki_psi, self.rate_kd_psi,
        #  self.att_kp_phi, self.att_ki_phi, self.att_kd_phi, self.att_kp_theta, self.att_ki_theta, self.att_kd_theta,self.att_kp_psi, self.att_ki_psi, self.att_kd_psi,) = self.pid_params

        (self.Kp_z, self.Ki_z, self.Kd_z, self.Kp_vz, self.Ki_vz, self.Kd_vz) = self.pid_params

        # print(f"kp_z: {self.Kp_z}, ki_z: {self.Ki_z}, kd_z: {self.Kd_z}")
        
        # Extract state variables
        x, y, z, dx, dy, dz, phi, theta, psi, p, q, r = state
        x_des, y_des, z_des = self.desired_position.get_position(current_time)
        vx_des, vy_des, vz_des = self.desired_velocity
        
        self.z_des_list.append(z_des)

        # Outer loop: Position control
        error_x = x_des - x
        error_y = y_des - y
        error_z = z_des - z
        print(f"error_z: {error_z}")

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
        print("az_des: ", az_des)
        
        u_f = np.clip(self.mass * (-self.g + az_des), 0, 200)  # 限制升力0-20N
        tau_phi = np.clip(tau_phi, -5, 5)
        tau_theta = np.clip(tau_theta, -5,5)
        tau_psi = np.clip(tau_psi, -5,5)

        return [u_f, tau_phi, tau_theta, tau_psi]
    
    def get_des_list(self):
        return self.z_des_list