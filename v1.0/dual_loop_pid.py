import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D
import configparser
import csv

class DualLoopPIDController:
    """
    双环 PID 控制器类，用于实现无人机的位置环和姿态环控制。

    外环（位置环）：根据当前位置与目标位置的误差，输出期望加速度，
      并利用小角度近似计算出期望的滚转角和俯仰角。

    内环（姿态环）：根据实际与期望的欧拉角误差，输出控制力矩。
    
    总升力由垂直方向 PID 输出计算得到：
      u_f = mass * (g + az_des)
    """
    def __init__(self, mass, gravity, desired_position, desired_yaw, dt,
                 kp_x=1.0, ki_x=0.0, kd_x=0.5,
                 kp_y=1.0, ki_y=0.0, kd_y=0.5,
                 kp_z=2.0, ki_z=0.0, kd_z=1.0,
                 kp_phi=5.0, ki_phi=0.0, kd_phi=2.0,
                 kp_theta=5.0, ki_theta=0.0, kd_theta=2.0,
                 kp_psi=1.0, ki_psi=0.0, kd_psi=0.2):
        self.mass = mass
        self.g = gravity
        self.desired_position = desired_position  # (x_des, y_des, z_des)
        self.desired_yaw = desired_yaw
        self.dt = dt
        
        # 外环 PID 参数
        self.Kp_x = kp_x; self.Ki_x = ki_x; self.Kd_x = kd_x
        self.Kp_y = kp_y; self.Ki_y = ki_y; self.Kd_y = kd_y
        self.Kp_z = kp_z; self.Ki_z = ki_z; self.Kd_z = kd_z
        
        # 内环 PID 参数
        self.Kp_phi = kp_phi; self.Ki_phi = ki_phi; self.Kd_phi = kd_phi
        self.Kp_theta = kp_theta; self.Ki_theta = ki_theta; self.Kd_theta = kd_theta
        self.Kp_psi = kp_psi; self.Ki_psi = ki_psi; self.Kd_psi = kd_psi
        
        # 外环积分项和前一时刻误差
        self.int_x = 0.0; self.last_error_x = 0.0
        self.int_y = 0.0; self.last_error_y = 0.0
        self.int_z = 0.0; self.last_error_z = 0.0
        # 内环积分项和前一时刻误差
        self.int_phi = 0.0; self.last_error_phi = 0.0
        self.int_theta = 0.0; self.last_error_theta = 0.0
        self.int_psi = 0.0; self.last_error_psi = 0.0
        
        self.last_time = None
        
    def update(self, current_time, state):
        """
        根据当前时间和状态计算新的控制输入。

        参数:
          current_time: 当前时间
          state: 状态向量 [x, y, z, dx, dy, dz, phi, theta, psi, p, q, r]

        返回:
          forces: 新的控制输入 [lift_force, tau_phi, tau_theta, tau_psi]
        """
        if self.last_time is None:
            dt = self.dt
        else:
            dt = current_time - self.last_time
        self.last_time = current_time

        # 提取状态
        x, y, z, dx, dy, dz, phi, theta, psi, p, q, r = state
        x_des, y_des, z_des = self.desired_position

        # 外环：位置误差
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

        # 外环 PID 输出期望加速度（m/s²）
        ax_des = self.Kp_x * error_x + self.Ki_x * self.int_x + self.Kd_x * d_error_x
        ay_des = self.Kp_y * error_y + self.Ki_y * self.int_y + self.Kd_y * d_error_y
        az_des = self.Kp_z * error_z + self.Ki_z * self.int_z + self.Kd_z * d_error_z

        # 利用小角度近似计算期望欧拉角
        phi_des = (1.0 / self.g) * ay_des
        theta_des = - (1.0 / self.g) * ax_des
        psi_des = self.desired_yaw

        # 内环：姿态误差
        error_phi = phi_des - phi
        error_theta = theta_des - theta
        error_psi = psi_des - psi

        self.int_phi += error_phi * dt
        self.int_theta += error_theta * dt
        self.int_psi += error_psi * dt

        d_error_phi = (error_phi - self.last_error_phi) / dt
        d_error_theta = (error_theta - self.last_error_theta) / dt
        d_error_psi = (error_psi - self.last_error_psi) / dt

        self.last_error_phi = error_phi
        self.last_error_theta = error_theta
        self.last_error_psi = error_psi

        # 内环 PID 输出控制力矩
        tau_phi = self.Kp_phi * error_phi + self.Ki_phi * self.int_phi + self.Kd_phi * d_error_phi
        tau_theta = self.Kp_theta * error_theta + self.Ki_theta * self.int_theta + self.Kd_theta * d_error_theta
        tau_psi = self.Kp_psi * error_psi + self.Ki_psi * self.int_psi + self.Kd_psi * d_error_psi

        # 总升力计算
        u_f = self.mass * (self.g + az_des)

        return [u_f, tau_phi, tau_theta, tau_psi]

########################################################################
# 全局 PID 控制器实例（由主函数创建并赋值）
########################################################################
pid_controller = None

def pid_callback(current_time, current_state, current_forces):
    """
    回调函数，在每个时间步积分后调用，通过全局 pid_controller 更新控制输入。
    """
    global pid_controller
    if pid_controller is None:
        raise ValueError("pid_controller 尚未初始化，请在主函数中创建并赋值。")
    new_forces = pid_controller.update(current_time, current_state)
    return new_forces