import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D
import configparser
import csv
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

from RK4Integrator import RK4Integrator

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
        x, y, z, dx, dy, dz, phi, theta, psi, p, q, r = state
        u_f, tau_phi, tau_theta, tau_psi = forces

        ddx = (1 / self.m) * ((np.cos(phi)*np.cos(theta)*np.sin(theta)*u_f) +
                              np.sin(phi)*np.sin(psi)*u_f - self.k_t*dx)
        ddy = (1 / self.m) * ((np.cos(phi)*np.sin(theta)*np.sin(psi) -
                              np.cos(psi)*np.sin(phi))*u_f - self.k_t*dy)
        ddz = (1 / self.m) * (np.cos(phi)*np.cos(theta)*u_f - self.m*self.g - self.k_t*dz)
        if z <= 0 and u_f < self.m*self.g:
            dz = 0
            ddz = 0
        dp = (1/self.Ix)*(-self.k_r*p - q*r*(self.Iz-self.Iy) + tau_phi)
        dq = (1/self.Iy)*(-self.k_r*q - r*p*(self.Ix-self.Iz) + tau_theta)
        dr = (1/self.Iz)*(-self.k_r*r - p*q*(self.Iy-self.Ix) + tau_psi)
        dphi = p + np.sin(phi)*np.tan(theta)*q + np.cos(phi)*np.tan(theta)*r
        dtheta = np.cos(phi)*q - np.sin(phi)*r
        dpsi = (1/np.cos(theta))*(np.sin(phi)*q + np.cos(phi)*r)
        return [dx, dy, dz, ddx, ddy, ddz, dphi, dtheta, dpsi, dp, dq, dr]

    def normalize_euler_angles(self, phi, theta, psi):
        phi = (phi + np.pi) % (2*np.pi) - np.pi
        theta = (theta + np.pi) % (2*np.pi) - np.pi
        psi = (psi + np.pi) % (2*np.pi) - np.pi
        return phi, theta, psi

    def simulate(self, initial_state, forces, time_span, time_eval, callback=None):
        integrator = RK4Integrator(self.rigid_body_dynamics, forces)
        times, states = integrator.integrate(time_eval, initial_state, callback)
        self.solution = type('Solution', (), {})()
        self.solution.y = states.T
        self.time_eval = times
        self.solution.y[6], self.solution.y[7], self.solution.y[8] = \
            self.normalize_euler_angles(self.solution.y[6], self.solution.y[7], self.solution.y[8])
        return times, states

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
        
        


########################################################################
# 强化学习可调PID控制器
########################################################################
class DualLoopPIDController:
    """
    四环PID控制器，所有参数可通过强化学习调整
    """
    def __init__(self, mass, gravity, desired_position, desired_velocity, desired_attitude, dt):
        self.mass = mass
        self.g = gravity
        self.desired_position = desired_position
        self.desired_velocity = desired_velocity
        self.desired_attitude = desired_attitude
        self.dt = dt

        # 初始化PID参数容器
        self.pid_params = np.zeros(36)
        
        # 初始化积分项和误差项
        self._init_integrators()
        
    def _init_integrators(self):
        # 初始化所有积分项和误差记录
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

    def set_pid_params(self, params):
        """将输入参数映射到合理范围"""
        # 使用sigmoid缩放比例项到(0, 10)
        self.pid_params[:18] = 10 * torch.sigmoid(torch.tensor(params[:18]))  
        # 使用tanh缩放微分项到(-5,5)
        self.pid_params[18:] = 5 * torch.tanh(torch.tensor(params[18:]))

    def update(self, current_time, state):
        # 解包PID参数
        (kp_x, ki_x, kd_x, kp_y, ki_y, kd_y, kp_z, ki_z, kd_z,
         kp_vx, ki_vx, kd_vx, kp_vy, ki_vy, kd_vy, kp_vz, ki_vz, kd_vz,
         att_kp_phi, att_ki_phi, att_kd_phi, att_kp_theta, att_ki_theta, att_kd_theta,
         att_kp_psi, att_ki_psi, att_kd_psi,
         rate_kp_phi, rate_ki_phi, rate_kd_phi, rate_kp_theta, rate_ki_theta, rate_kd_theta,
         rate_kp_psi, rate_ki_psi, rate_kd_psi) = self.pid_params

        # 计算时间步长
        dt = self.dt if self.last_time is None else current_time - self.last_time
        self.last_time = current_time

        # 状态解包
        x, y, z, dx, dy, dz, phi, theta, psi, p, q, r = state
        x_des, y_des, z_des = self.desired_position
        vx_des, vy_des, vz_des = self.desired_velocity

        # 外环：位置控制
        error_x = x_des - x
        error_y = y_des - y
        error_z = z_des - z
        
        self.int_x += error_x * dt
        self.int_y += error_y * dt
        self.int_z += error_z * dt
        
        d_error_x = (error_x - self.last_error_x) / dt
        d_error_y = (error_y - self.last_error_y) / dt
        d_error_z = (error_z - self.last_error_z) / dt
        
        vx_des = kp_x * error_x + ki_x * self.int_x + kd_x * d_error_x
        vy_des = kp_y * error_y + ki_y * self.int_y + kd_y * d_error_y
        vz_des = kp_z * error_z + ki_z * self.int_z + kd_z * d_error_z

        # 速度环控制
        error_vx = vx_des - dx
        error_vy = vy_des - dy
        error_vz = vz_des - dz
        
        self.int_vx += error_vx * dt
        self.int_vy += error_vy * dt
        self.int_vz += error_vz * dt
        
        d_error_vx = (error_vx - self.last_error_vx) / dt
        d_error_vy = (error_vy - self.last_error_vy) / dt
        d_error_vz = (error_vz - self.last_error_vz) / dt
        
        ax_des = kp_vx * error_vx + ki_vx * self.int_vx + kd_vx * d_error_vx
        ay_des = kp_vy * error_vy + ki_vy * self.int_vy + kd_vy * d_error_vy
        az_des = kp_vz * error_vz + ki_vz * self.int_vz + kd_vz * d_error_vz

        # 期望姿态计算
        phi_des = (ay_des / self.g)
        theta_des = -(ax_des / self.g)
        psi_des = self.desired_attitude[2]

        # 姿态环控制
        error_phi = phi_des - phi
        error_theta = theta_des - theta
        error_psi = psi_des - psi
        
        self.int_phi_att += error_phi * dt
        self.int_theta_att += error_theta * dt
        self.int_psi_att += error_psi * dt
        
        d_error_phi = (error_phi - self.last_error_phi_att) / dt
        d_error_theta = (error_theta - self.last_error_theta_att) / dt
        d_error_psi = (error_psi - self.last_error_psi_att) / dt
        
        p_des = att_kp_phi * error_phi + att_ki_phi * self.int_phi_att + att_kd_phi * d_error_phi
        q_des = att_kp_theta * error_theta + att_ki_theta * self.int_theta_att + att_kd_theta * d_error_theta
        r_des = att_kp_psi * error_psi + att_ki_psi * self.int_psi_att + att_kd_psi * d_error_psi

        # 角速度环控制
        error_p = p_des - p
        error_q = q_des - q
        error_r = r_des - r
        
        self.int_p += error_p * dt
        self.int_q += error_q * dt
        self.int_r += error_r * dt
        
        d_error_p = (error_p - self.last_error_p) / dt
        d_error_q = (error_q - self.last_error_q) / dt
        d_error_r = (error_r - self.last_error_r) / dt
        
        tau_phi = rate_kp_phi * error_p + rate_ki_phi * self.int_p + rate_kd_phi * d_error_p
        tau_theta = rate_kp_theta * error_q + rate_ki_theta * self.int_q + rate_kd_theta * d_error_q
        tau_psi = rate_kp_psi * error_r + rate_ki_psi * self.int_r + rate_kd_psi * d_error_r

        # 升力计算
        u_f = self.mass * (self.g + az_des)
        
        # 添加控制量限幅
        u_f = np.clip(self.mass * (self.g + az_des), 0, 200)  # 限制升力0-20N
        tau_phi = np.clip(tau_phi, -5, 5)
        tau_theta = np.clip(tau_theta, -5,5)
        tau_psi = np.clip(tau_psi, -5,5)
        
        return [u_f, tau_phi, tau_theta, tau_psi]
    
    

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
                new_forces = callback(time_eval[idx+1], new_state, self.forces)  # Update forces using PID callback
                if new_forces is not None:
                    self.forces = new_forces
            state = new_state
        self.states.append(state.copy())
        return time_eval, np.array(self.states)
    
    
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
        
        # print("pid_kp_z: {}, pid_ki_z: {}, pid_kd_z: {}".format(self.pid_controller.Kp_z, self.pid_controller.Ki_z, self.pid_controller.Kd_z))
        # print("current_state: {}, current_forces: {}".format(current_state, current_forces))

        new_forces = self.pid_controller.update(current_time, current_state)  # Update the PID controller with the current state and time
        return new_forces
    
    
########################################################################
# 强化学习环境
########################################################################
class QuadrotorEnv(gym.Env):
    def __init__(self, initial_state, target_position, target_attitude):
        super(QuadrotorEnv, self).__init__()
        
        # 动作空间：36个PID参数，范围[0, 20]
        self.action_space = spaces.Box(
            low=0.0,
            high=200.0,
            shape=(36,),
            dtype=np.float32
        )

        # 观测空间
        self.observation_space = spaces.Box(
            low=np.array([-np.inf]*12),
            high=np.array([np.inf]*12),
            dtype=np.float32
        )

        # 初始化无人机模型
        self.drone = DroneSimulation(
            mass=3.18,
            inertia=[0.1, 0.1, 0.1],
            drag_coeffs=[0.1, 0.1],
            gravity=9.81
        )
        
        # 初始化PID控制器
        self.pid = DualLoopPIDController(
            mass=3.18,
            gravity=9.81,
            desired_position=target_position,
            desired_velocity=[0,0,0],
            desired_attitude=target_attitude,
            dt=0.01
        )
        
        # 初始状态参数
        self.initial_state = initial_state
        self.state = initial_state.copy()
        self.step_count = 0
        
        # 添加观测标准化参数
        self.obs_mean = np.zeros(12)
        self.obs_std = np.ones(12)
        
    def _normalize_obs(self, obs):
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)

    def reset(self, seed=None, **kwargs):
        # 如果传入了seed参数，则使用它来初始化随机数生成器
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.state = np.zeros(12)  # Reset state to initial values
        self.done = False
        return self.state, {}

    def step(self, action):
        # 设置PID参数
        self.pid.set_pid_params(action)
        
        # 运行控制周期
        control = self.pid.update(self.step_count*0.01, self.state)
        time_eval = np.linspace(0, 0.01, 2)
        _, new_states = self.drone.simulate(
            self.state, control, 
            time_span=(0,0.01),
            time_eval=time_eval
        )
        
        # 更新状态
        self.state = new_states[-1]
        self.step_count += 1
        
        # 在状态更新后添加数值检查
        if np.any(np.isnan(self.state)):
            print(f"Invalid state detected at step {self.step_count}!")
            self.state = np.nan_to_num(self.state)  # 自动替换NaN为0
            reward = -1000  # 给予极大惩罚
            done = True
        
        # 计算奖励
        pos_error = np.linalg.norm(self.state[:3] - self.pid.desired_position)
        att_error = np.linalg.norm(self.state[6:9] - self.pid.desired_attitude)
        reward = - (pos_error + 0.1*att_error)
        
        # 终止条件
        done = pos_error < 0.1 or self.step_count >= 500
        
        return self._normalize_obs(self.state), reward, done, False, {}
        # return self.state, reward, terminated, {}, {}

    def render(self):
        """渲染环境状态"""
        # print(f"Step: {self.step_count}, State: {self.state}")

    def close(self):
        """关闭环境"""
        pass
    
class MyCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MyCallback, self).__init__(verbose)

    def _on_step(self):
        # 在每次训练步之后打印信息
        # print("已执行步数:", self.num_timesteps)
        return True



    
########################################################################
# 训练和测试
########################################################################
if __name__ == "__main__":
    # 初始化环境参数
    initial_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    target_position = [0, 0, 5]
    target_attitude = [0, 0, 0]

    # 创建环境
    env = QuadrotorEnv(initial_state, target_position, target_attitude)
    
    # 创建PPO模型
    model = PPO("MlpPolicy", env, verbose=1,
            policy_kwargs=dict(
                net_arch=dict(pi=[128,128], vf=[128,128]),  # 缩小网络规模
                activation_fn=torch.nn.Tanh),  # 添加tanh激活函数
            learning_rate=1e-4,  # 降低学习率
            clip_range=0.2,  # 使用默认PPO裁剪范围
            max_grad_norm=0.5,  # 添加梯度裁剪
            device='cpu')
    
    # 训练模型
    model.learn(total_timesteps=1e5)
    
    # 测试训练结果
    obs, _ = env.reset()
    for _ in range(500):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        if done:
            break
    print("Final position:", obs[:3])


# # 使用PID控制器与四旋翼仿真环境结合
# if __name__ == "__main__":
#     # 初始化PID控制器
#     pid_controller = DualLoopPIDController(
#         mass=1.0,
#         gravity=9.81,
#         desired_position=[0, 0, 10],  # 目标位置
#         desired_velocity=[0, 0, 0],   # 目标速度
#         desired_attitude=[0, 0, 0],   # 目标姿态
#         dt=0.1
#     )

#     # 初始化四旋翼环境
#     env = QuadrotorEnv(
#         mass=3.18,
#         inertia=[0.029618, 0.069585, 0.042503],  # 假设惯性矩阵
#         drag_coeffs=[0.0, 0.0],      # 假设阻力系数
#         gravity=9.81,                 # 重力加速度
#         pid_controller=pid_controller
#     )

#     # 运行环境并训练
#     state = env.reset()
#     done = False
#     while not done:
#         action = env.action_space.sample()  # 随机选择动作
#         next_state, reward, done, _, _ = env.step(action)
#         print(done)
#         env.render()




# def register_quadrotor_env():
#     gym.envs.registration.register(
#         id='Quadrotor-v0',
#         entry_point='__main__:QuadrotorEnv',
#         max_episode_steps=10,
#     )




















# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_checker import check_env
# import matplotlib.pyplot as plt

# class DroneSimulation:
#     def __init__(self, mass, inertia, drag_coeffs, gravity):
#         self.m = mass
#         self.Ix, self.Iy, self.Iz = inertia
#         self.J = np.diag([self.Ix, self.Iy, self.Iz])
#         self.k_t, self.k_r = drag_coeffs
#         self.g = gravity

#     def rigid_body_dynamics(self, t, state, forces):
#         x, y, z, dx, dy, dz, phi, theta, psi, p, q, r = state
#         u_f, tau_phi, tau_theta, tau_psi = forces
#         ddx = (1 / self.m) * (u_f * np.cos(phi) * np.cos(theta) - self.k_t * dx)
#         ddy = (1 / self.m) * (u_f * np.sin(phi) * np.sin(theta) - self.k_t * dy)
#         ddz = (1 / self.m) * (u_f - self.m * self.g - self.k_t * dz)
#         dp = (1 / self.Ix) * (-self.k_r * p + tau_phi)
#         dq = (1 / self.Iy) * (-self.k_r * q + tau_theta)
#         dr = (1 / self.Iz) * (-self.k_r * r + tau_psi)
#         dphi = p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r
#         dtheta = np.cos(phi) * q - np.sin(phi) * r
#         dpsi = (1 / np.cos(theta)) * (np.sin(phi) * q + np.cos(phi) * r)
#         return [dx, dy, dz, ddx, ddy, ddz, dphi, dtheta, dpsi, dp, dq, dr]

#     def normalize_euler_angles(self, phi, theta, psi):
#         return ((phi + np.pi) % (2 * np.pi) - np.pi, (theta + np.pi) % (2 * np.pi) - np.pi, (psi + np.pi) % (2 * np.pi) - np.pi)

#     def simulate(self, initial_state, forces, time_span, time_eval):
#         integrator = RK4Integrator(self.rigid_body_dynamics, forces)
#         times, states = integrator.integrate(time_eval, initial_state)
#         self.solution = type('Solution', (), {})()
#         self.solution.y = states.T
#         self.time_eval = times
#         self.solution.y[6], self.solution.y[7], self.solution.y[8] = self.normalize_euler_angles(self.solution.y[6], self.solution.y[7], self.solution.y[8])
#         return times, states
    
# ########################################################################
# # Fourth-order Runge-Kutta Integrator Class (with callback)
# ########################################################################
# class RK4Integrator:
#     """
#     Fourth-order Runge-Kutta integrator.
#     After each integration step, calls a callback function to update the control input.
#     """
#     def __init__(self, func, forces):
#         self.func = func
#         self.forces = forces
#         self.states = []

#     def integrate(self, time_eval, initial_state, callback=None):
#         dt = time_eval[1] - time_eval[0]
#         state = np.array(initial_state)
#         self.states = []
#         for idx in range(len(time_eval) - 1):
#             self.states.append(state.copy())
#             t_current = time_eval[idx]
#             k1 = np.array(self.func(t_current, state, self.forces))
#             k2 = np.array(self.func(t_current + dt/2, state + dt/2 * k1, self.forces))
#             k3 = np.array(self.func(t_current + dt/2, state + dt/2 * k2, self.forces))
#             k4 = np.array(self.func(t_current + dt, state + dt * k3, self.forces))
#             new_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
#             if new_state[2] < 0:
#                 new_state[2] = 0
#                 new_state[5] = 0
#             if callback is not None:
#                 new_forces = callback(time_eval[idx+1], new_state, self.forces)  # Update forces using PID callback
#                 if new_forces is not None:
#                     self.forces = new_forces
#             state = new_state
#         self.states.append(state.copy())
#         return time_eval, np.array(self.states)

# class DualLoopPIDController:
#     def __init__(self, mass, gravity, desired_position, dt, kp_x, ki_x, kd_x, kp_y, ki_y, kd_y, kp_z, ki_z, kd_z):
#         self.mass = mass
#         self.g = gravity
#         self.desired_position = desired_position
#         self.dt = dt
#         self.Kp_x, self.Ki_x, self.Kd_x = kp_x, ki_x, kd_x
#         self.Kp_y, self.Ki_y, self.Kd_y = kp_y, ki_y, kd_y
#         self.Kp_z, self.Ki_z, self.Kd_z = kp_z, ki_z, kd_z
#         self.int_x, self.last_error_x = 0.0, 0.0
#         self.int_y, self.last_error_y = 0.0, 0.0
#         self.int_z, self.last_error_z = 0.0, 0.0
#         self.last_time = None

#     def update(self, current_time, state):
#         if self.last_time is None:
#             dt = self.dt
#         else:
#             dt = current_time - self.last_time
#         self.last_time = current_time

#         x, y, z, dx, dy, dz, phi, theta, psi, p, q, r = state
#         x_des, y_des, z_des = self.desired_position

#         # Position control
#         error_x = x_des - x
#         error_y = y_des - y
#         error_z = z_des - z

#         self.int_x += error_x * dt
#         self.int_y += error_y * dt
#         self.int_z += error_z * dt

#         d_error_x = (error_x - self.last_error_x) / dt
#         d_error_y = (error_y - self.last_error_y) / dt
#         d_error_z = (error_z - self.last_error_z) / dt

#         self.last_error_x, self.last_error_y, self.last_error_z = error_x, error_y, error_z

#         vx_des = self.Kp_x * error_x + self.Ki_x * self.int_x + self.Kd_x * d_error_x
#         vy_des = self.Kp_y * error_y + self.Ki_y * self.int_y + self.Kd_y * d_error_y
#         vz_des = self.Kp_z * error_z + self.Ki_z * self.int_z + self.Kd_z * d_error_z

#         # Output control forces
#         u_f = self.mass * (-self.g + vz_des)
#         tau_phi, tau_theta, tau_psi = 0, 0, 0  # Assume simple control for now

#         return [u_f, tau_phi, tau_theta, tau_psi]

# class QuadrotorEnv(gym.Env):
#     def __init__(self, mass, inertia, drag_coeffs, gravity, pid_controller):
#         super(QuadrotorEnv, self).__init__()
#         self.action_space = spaces.Box(low=np.array([0.0, -10.0, -10.0, -10.0]), high=np.array([10.0, 10.0, 10.0, 10.0]), dtype=np.float32)
#         self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, 0, -np.inf, -np.inf, -np.inf, -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf], dtype=np.float32), high=np.array([np.inf, np.inf, 10, np.inf, np.inf, np.inf, np.pi, np.pi, np.pi, np.inf, np.inf, np.inf], dtype=np.float32), dtype=np.float32)
#         self.pid_controller = pid_controller
#         self.drone_sim = DroneSimulation(mass, inertia, drag_coeffs, gravity)
#         self.state = np.zeros(12)
#         self.done = False
#         self.step_count = 0

#     def reset(self, seed=None, **kwargs):
#         # 如果传入了seed参数，则使用它来初始化随机数生成器
#         self.np_random, seed = gym.utils.seeding.np_random(seed)
#         self.state = np.zeros(12)  # Reset state to initial values
#         self.done = False
#         return self.state, {}

#     def step(self, action):
#         forces = self.pid_controller.update(self.step_count, self.state)
#         t = self.step_count
#         state = self.state
#         integrator = RK4Integrator(self.drone_sim.rigid_body_dynamics, forces)
#         time_eval = np.linspace(0, 0.1, 100)
#         self.times, self.state = self.drone_sim.simulate(state, forces, time_span=(0, 10), time_eval=time_eval)
#         self.state = self.state[-1]
#         terminated = np.linalg.norm(self.state[0:3]) > 10
#         self.step_count += 1
#         reward = -np.linalg.norm(self.state[0:3])
#         return self.state, reward, terminated, False, {}

#     def render(self):
#         print(f"Step: {self.step_count}, State: {self.state}")

#     def close(self):
#         pass

# # 使用 PID 控制器与四旋翼仿真环境结合
# if __name__ == "__main__":
#     pid_controller = DualLoopPIDController(
#         mass=3.18,
#         gravity=9.81,
#         desired_position=[5, 5, 10],
#         dt=0.1,
#         kp_x=1.5, ki_x=0.0, kd_x=0.5,
#         kp_y=1.5, ki_y=0.0, kd_y=0.5,
#         kp_z=2.5, ki_z=0.0, kd_z=1.0
#     )

#     env = QuadrotorEnv(
#         mass=3.18,
#         inertia=[0.029618, 0.069585, 0.042503],
#         drag_coeffs=[0.0, 0.0],
#         gravity=9.81,
#         pid_controller=pid_controller
#     )

#     model = PPO("MlpPolicy", env, verbose=1)
#     model.learn(total_timesteps=10000)

#     # Test the trained model
#     state, _ = env.reset()
#     done = False
#     while not done:
#         action, _ = model.predict(state)
#         state, reward, done, _, _ = env.step(action)
#         env.render()
