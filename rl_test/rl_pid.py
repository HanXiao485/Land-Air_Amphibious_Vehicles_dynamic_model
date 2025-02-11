import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

from drone_simulation import DroneSimulation
from dual_loop_pid import DualLoopPIDController
from RK4Integrator import RK4Integrator
from call_back import PIDCallbackHandler


class QuadrotorEnv(gym.Env):
    def __init__(self, mass, inertia, drag_coeffs, gravity, pid_controller):
        super(QuadrotorEnv, self).__init__()

        # 动作空间：升力u_f，三个力矩tau_phi, tau_theta, tau_psi
        self.action_space = spaces.Box(
            low=np.array([0.0, -10.0, -10.0, -10.0]),
            high=np.array([10.0, 10.0, 10.0, 10.0]),
            dtype=np.float32
        )

        # 观测空间：位置（x, y, z），速度（dx, dy, dz），姿态角（phi, theta, psi），角速度（p, q, r）
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, 0, -np.inf, -np.inf, -np.inf, -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, 10, np.inf, np.inf, np.inf, np.pi, np.pi, np.pi, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )

        # 初始化PID控制器和四旋翼动力学模型
        self.pid_controller = pid_controller
        self.drone_sim = DroneSimulation(mass, inertia, drag_coeffs, gravity)

        # 初始状态（例如：初始为静止状态）
        self.state = np.zeros(12)
        self.done = False
        
        # 初始化计数器
        self.step_count = 0

    def reset(self):
        """重置环境状态"""
        self.state = np.zeros(12)  # 初始状态为零
        self.done = False
        return self.state

    def step(self, action):
        """根据动作更新环境状态"""
        # 使用PID控制器根据当前状态计算控制输入

        u_f, tau_phi, tau_theta, tau_psi = self.pid_controller.update(
            current_time=self.step_count, state=self.state
        )

        # 将PID控制器生成的控制输入传递给四旋翼动力学模型
        forces = [u_f, tau_phi, tau_theta, tau_psi]
        t = self.step_count  # 当前时间步
        state = self.state

        # 使用RK4积分器计算状态更新
        callback_handler = PIDCallbackHandler(pid_controller)
        integrator = RK4Integrator(self.drone_sim.rigid_body_dynamics, forces)
        time_eval = np.linspace(0, 0.1, 10)  # 仿真时间步长
        self.drone_sim.simulate(state, forces, time_span=(0, 10), time_eval=time_eval, callback=callback_handler.callback)

        # 获取新的状态
        self.state = self.drone_sim.data_results()

        # 判断任务是否完成
        terminated = np.linalg.norm(self.state[0:3]) > 10  # 假设任务完成时超出10米
        self.step_count += 1

        # 奖励函数：简单的惩罚当前位置越远
        reward = -np.linalg.norm(self.state[0:3])

        return self.state, reward, terminated, False, {}

    def render(self):
        """渲染环境状态"""
        print(f"Step: {self.step_count}, State: {self.state}")

    def close(self):
        """关闭环境"""
        pass

# 使用PID控制器与四旋翼仿真环境结合
if __name__ == "__main__":
    # 初始化PID控制器
    pid_controller = DualLoopPIDController(
        mass=1.0,
        gravity=9.81,
        desired_position=[0, 0, 10],  # 目标位置
        desired_velocity=[0, 0, 0],   # 目标速度
        desired_attitude=[0, 0, 0],   # 目标姿态
        dt=0.1
    )

    # 初始化四旋翼环境
    env = QuadrotorEnv(
        mass=3.18,
        inertia=[0.029618, 0.069585, 0.042503],  # 假设惯性矩阵
        drag_coeffs=[0.0, 0.0],      # 假设阻力系数
        gravity=9.81,                 # 重力加速度
        pid_controller=pid_controller
    )

    # 运行环境并训练
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # 随机选择动作
        next_state, reward, done, _, _ = env.step(action)
        print(done)
        env.render()




def register_quadrotor_env():
    gym.envs.registration.register(
        id='Quadrotor-v0',
        entry_point='__main__:QuadrotorEnv',
        max_episode_steps=10,
    )