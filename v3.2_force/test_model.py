import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3, A2C, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
import matplotlib.pyplot as plt

from drone_simulation import DroneSimulation
from dual_loop_pid import DualLoopPIDController
from RK4Integrator import RK4Integrator
from call_back import PIDCallbackHandler
from stable_baselines3.common.callbacks import BaseCallback
from curve import Curve


class QuadrotorEnv(gym.Env):
    def __init__(self, mass, inertia, drag_coeffs, gravity, pid_params):
        super(QuadrotorEnv, self).__init__()

        # # 动作空间：pid参数，四环共36个
        # self.action_space = spaces.Box(
        #     low=np.array([0.1]*36),
        #     high=np.array([6.0]*36),
        #     dtype=np.float32
        # )
        
        # 动作空间：参数，z方向共6个
        self.action_space = spaces.Box(
            low=np.array([0.0]*1, dtype=np.float32),
            high=np.array([200.0]*1, dtype=np.float32),
            dtype=np.float32
        )

        # 观测空间：位置（x, y, z），速度（dx, dy, dz），姿态角（phi, theta, psi），角速度（p, q, r）
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.pi, np.pi, np.pi, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        
        self.curve = Curve(a=0.0, b=0.0, c=0.0, d=0.0, e=1.0, w=0.0)

        # # 初始化PID控制器和四旋翼动力学模型
        # self.pid_controller = DualLoopPIDController(
        #     mass=pid_params['mass'],
        #     gravity=pid_params['gravity'],
        #     desired_position = self.curve,
        #     desired_velocity=pid_params['desired_velocity'],
        #     desired_attitude=pid_params['desired_attitude'],
        #     dt=pid_params['dt']
        # )
        self.drone_sim = DroneSimulation(mass, inertia, drag_coeffs, gravity)

        # 初始状态（例如：初始为静止状态）
        self.state = np.zeros(12)
        self.done = False
        
        # 初始化计数器
        self.step_count = 0
        
        self.obs_mean = np.zeros(12)
        self.obs_std = np.ones(12)
        
    def _normalize_obs(self, obs):
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)

    def reset(self, seed=None, **kwargs):
        """重置环境状态"""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        # self.state = np.zeros(12)  # 初始状态为零
        self.t = np.random.randint(0, 2000)
        self.current_time = self.t
        h = self.curve.get_position(self.t)[2]
        self.state = np.array([0.0, 0.0, h, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.done = False
        self.done_state = []
        self.reward_state = []
        self.state_list = []
        self.z_des_list = []
        
        self.curve = Curve(a=0.0, b=0.0, c=0.0, d=0.0, e=1.0, w=0.0)
        # self.pid_controller = DualLoopPIDController(
        #     mass=pid_params['mass'],
        #     gravity=pid_params['gravity'],
        #     desired_position = self.curve,
        #     desired_velocity=pid_params['desired_velocity'],
        #     desired_attitude=pid_params['desired_attitude'],
        #     dt=pid_params['dt']
        # )
        return self.state, {}

    def step(self, action):
        """根据动作更新环境状态"""
        
        u_f = action[0]
        # print("======================================================================================")
        
        # 获取PID控制器生成的期望值，一次update后生成一个期望值
        self.z_des_list.append(self.curve.get_position(self.current_time + 1)[2])

        # 将PID控制器生成的控制输入传递给四旋翼动力学模型
        forces = [u_f, 0, 0, 0]
        state = self.state

        # 使用RK4积分器计算状态更新
        integrator = RK4Integrator(self.drone_sim.rigid_body_dynamics, forces)
        time_eval = np.linspace(0, 0.1, 100)  # 仿真时间步长
        self.times, self.state = self.drone_sim.simulate(state, forces, time_span=(0, 10), time_eval=time_eval, callback = None)  # self.state为10*12的矩阵，10为时间步数
        
        # 上一个仿真时间段内，每一个时间步后的仿真结果
        self.state_list.append(self.state[:, 2][-1])
        
        # # 判断加速度
        # if self.t > 0:
        #     print(f"error = {self.state_list[-1] - self.state_list[-2]}")
            
        # 获取新的状态,上一个仿真时间段内最后一个时间步的输出状态
        self.reward_state.append(self.state[:, 2])
        self.state = self.state[-1]

        # 判断任务是否完成
        self.done_state.append(self.state[2])
        
        # if self.state[2] > 4 or self.state[2] < 0.5:
        #     terminated = True
        if self.current_time > 2000:
            terminated = True
        elif self.state[2] < 20 or self.state[2] > 80:
            terminated = True
        else:
            terminated = False
            
        def reward_function(state, des_list, action ,current_time):
            # 奖励函数：简单的惩罚当前位置越远
            mean = np.mean(abs(np.array(state) - np.array(des_list)))
            var = np.var(np.array(state) - np.array(des_list), ddof=0)
            
            traget_error = des_list[-1] - state[-1]
            
            reward = - np.abs(traget_error)
            
            if traget_error > 0 and action > 0:
                reward += 0
            elif traget_error < 0 and action > 0:
                reward -= 100

            return reward
        
        # terminated = np.linalg.norm(self.state[0:3]) > 10  # 假设任务完成时超出10米
        
        reward = reward_function(self.state_list, self.z_des_list, u_f, self.current_time)
        # reward = -abs(self.state[2] - 5) - 0.1 * np.sum(np.square(self.state[3:6])) - 0.1 * np.sum(np.square(self.state[6:9]))
        # print(f"Step: {self.step_count}, State: {self.state[3]}, force: {u_f}, Reward: {reward}")
        
        # if terminated:
        #     print(f"time: {current_time}")
        #     print(f"high: {self.reward_state[-1][-1]}, \nforce: {u_f} \nreward: {reward}")
        #     print(f"kp_z: {action[0]}, ki_z: {action[1]}, kd_z: {action[2]}, kp_vx: {action[3]}, ki_vx: {action[4]}, kd_vx: {action[5]}")
        
        self.step_count += 1
        self.current_time += 1
        
        return self._normalize_obs(self.state), reward, terminated, False, {}

    def render(self):
        """渲染环境状态"""
        # print(f"Step: {self.step_count}, State: {self.state}")

    def close(self):
        """关闭环境"""
        pass



# 使用PID控制器与四旋翼仿真环境结合
if __name__ == "__main__":
   # 创建多个环境实例
    def make_env():
        def _init():
            pid_params = {
                'mass': 3.18,
                'gravity': 9.81,
                'desired_position': [0, 0, 5],  # 目标位置
                'desired_velocity': [0, 0, 0],   # 目标速度
                'desired_attitude': [0, 0, 0],   # 目标姿态
                'dt': 0.1
            }

            env = QuadrotorEnv(
                mass=3.18,
                inertia=[0.029618, 0.069585, 0.042503],  # 假设惯性矩阵
                drag_coeffs=[0.0, 0.0],      # 假设阻力系数
                gravity=9.81,                 # 重力加速度
                pid_params=pid_params  # 将PID控制器的参数传递给环境
            )
            return env
        return _init

    # 创建多环境
    num_envs = 1  # 设置需要的环境数量
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    # # 配置TensorBoard日志
    # log_dir = "./tensorboard_logs/"
    # # log_dir = None
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    
    # model = PPO("MlpPolicy", env, verbose=1,
    #             n_steps=10000,
    #             batch_size=4096,
    #             n_epochs=50,
    #             policy_kwargs=dict(
    #                 net_arch=dict(pi=[128, 128, 128, 128], vf=[128, 128, 128, 128]),   # 缩小网络规模
    #                 activation_fn=torch.nn.ReLU),  # 添加tanh激活函数
    #             learning_rate=3e-3,  # 降低学习率
    #             use_sde=True,  # 使用Gaussian noise
    #             clip_range=0.2,  # 使用默认PPO裁剪范围
    #             max_grad_norm=0.5,  # 添加梯度裁剪
    #             device = device,
    #             tensorboard_log=log_dir)  # 将TensorBoard日志路径添加到模型配置中

    # # 训练模型
    # model.learn(total_timesteps=4e6,
    #             progress_bar=True)
    
    # # 保存训练后的模型
    # model.save("quadrotor_model")
    
    model = PPO.load("quadrotor_model_20250220_2_256256_1.28", env=env)
    
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    
    # print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    env = model.get_env()
    obs = env.reset()
    
    # 测试训练结果
    obs = env.reset()
    for _ in range(500):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        print("reward:", reward)
        if done:
            break
    print("reward:", reward)
    