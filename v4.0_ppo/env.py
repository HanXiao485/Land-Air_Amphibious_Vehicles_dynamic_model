import logging
import csv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import os
import time

from stable_baselines3 import PPO, SAC, TD3, A2C, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from ppo import Actor, Critic, ReplayMemory, PPOAgent
import matplotlib.pyplot as plt

from drone_simulation import DroneSimulation
from dual_loop_pid import DualLoopPIDController
from RK4Integrator import RK4Integrator
from call_back import PIDCallbackHandler
from stable_baselines3.common.callbacks import BaseCallback
from curve import Curve

from tensorboardX import SummaryWriter  # 用于TensorBoard记录

# 设置 logging 配置
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,  # 记录信息级别为 INFO，可以改为 DEBUG 以记录更多信息
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler("training_log.txt")  # 输出到文件
    ]
)
logger = logging.getLogger()

class QuadrotorEnv(gym.Env):
    def __init__(self, mass, inertia, drag_coeffs, gravity, pid_params):
        super(QuadrotorEnv, self).__init__()
        
        # 打开 CSV 文件并写入表头（如果文件不存在）
        self.csv_file = open("rewards.csv", mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        # 写入表头
        self.csv_writer.writerow(['TimeStep', 'Reward'])
        
        # 动作空间：参数，z方向共6个
        self.action_space = spaces.Box(
            low=np.array([0, -50, -50, -50], dtype=np.float32),
            high=np.array([200, 50, 50, 50], dtype=np.float32),
            dtype=np.float32
        )

        # 观测空间：位置（x, y, z），速度（dx, dy, dz），姿态角（phi, theta, psi），角速度（p, q, r）
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.pi, np.pi, np.pi, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        
        self.curve = Curve(a=1.0, b=1.0, c=0.0, d=0.0, e=1.0, w=1.0)

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
        x, y, z = self.curve.get_position(self.t)[0], self.curve.get_position(self.t)[1], self.curve.get_position(self.t)[2]
        self.state = np.array([x, y, z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.done = False
        self.done_state = []
        self.reward_state = []
        self.state_list = []
        self.des_list = []
        self.vel_list = np.array([0.0, 0.0, 0.0])
        self.att_list = np.array([0.0, 0.0, 0.0])
        self.n_steps = 64
        
        return self.state, {}

    def step(self, action):
        """根据动作更新环境状态"""
        
        u_f, tau_phi, tau_theta, tau_psi = action[0], action[1], action[2], action[3]
        # print("======================================================================================")
        
        # 获取PID控制器生成的期望值，一次update后生成一个期望值
        self.des_list.append(self.curve.get_position(self.current_time + 1))

        # 将PID控制器生成的控制输入传递给四旋翼动力学模型
        forces = [u_f, tau_phi, tau_theta, tau_psi]
        
        
        state = self.state

        # 使用RK4积分器计算状态更新
        integrator = RK4Integrator(self.drone_sim.rigid_body_dynamics, forces)
        time_eval = np.linspace(0, 0.1, 100)  # 仿真时间步长
        self.times, self.state = self.drone_sim.simulate(state, forces, time_span=(0, 10), time_eval=time_eval, callback = None)  # self.state为10*12的矩阵，10为时间步数
        
        # 上一个仿真时间段内，每一个时间步后的仿真结果 1*12
        self.state_list.append(self.state[:][-1])
        
            
        # 获取新的状态,上一个仿真时间段内最后一个时间步的输出状态 1*12
        self.state = self.state[-1]
        # self.state_end = self.state[-1]
        
        def reward_function(state, des_list, vel_list, att_list, action ,current_time):
            
            dis_error = np.linalg.norm(des_list[-1] - state[-1][0:3])  # x,y,z 1*3
            vel_error = np.linalg.norm(vel_list[-1] - state[-1][6:9])  # vx,vy,vz 1*3
            
            phi_error = np.linalg.norm(att_list[0] - state[-1][6])
            theta_error = np.linalg.norm(att_list[1] - state[-1][7])
            
            r_pose = np.exp(-dis_error * 1.2)
            r_phi = 0.5 / (1 + np.square(phi_error))
            r_theta = 0.5 / (1 + np.square(theta_error))
            
            reward = r_pose + r_pose * (r_phi + r_theta)
            
            done =  (current_time >= 2000) | (dis_error > 0.5)
            
            return reward, done
        
        reward, done = reward_function(self.state_list, self.des_list, self.vel_list, self.att_list, u_f, self.current_time)
        
        # 记录奖励和时间步
        self.csv_writer.writerow([self.current_time, reward])
        
        if self.current_time % self.n_steps == 0:
            logger.info(f"Step: {self.current_time}, Average reward: {reward:.8f}")
        
        self.step_count += 1
        self.current_time += 1
        
        return self._normalize_obs(self.state), reward, done, False, {}
        

    def render(self):
        """渲染环境状态"""

    def close(self):
        """关闭环境"""
        self.csv_file.close()
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

    # 配置TensorBoard日志
    log_dir = "./tensorboard_logs/"

    # 创建多环境
    num_envs = 1  # 设置需要的环境数量
    env = QuadrotorEnv(
        mass=3.18,
        inertia=[0.029618, 0.069585, 0.042503],  # 假设惯性矩阵
        drag_coeffs=[0.0, 0.0],      # 假设阻力系数
        gravity=9.81,                 # 重力加速度
        pid_params = {
                'mass': 3.18,
                'gravity': 9.81,
                'desired_position': [0, 0, 5],  # 目标位置
                'desired_velocity': [0, 0, 0],   # 目标速度
                'desired_attitude': [0, 0, 0],   # 目标姿态
                'dt': 0.1
            }
    )
    
    # Directory for saving trained models
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model = current_dir + "/models/"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    NUM_EPISODE = 3000
    NUM_STEP = 200
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.shape[0]
    # ACTION_DIM = env.action_space.n
    BATCH_SIZE = 25
    UPDATE_INTERVAL = 50

    agent = PPOAgent(STATE_DIM, ACTION_DIM, BATCH_SIZE)
    
    REWARD_BUFFER = np.empty(shape = NUM_EPISODE)
    best_reward = -2000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练!")
        agent.actor = torch.nn.DataParallel(agent.actor)  # 使用DataParallel进行多GPU训练
        agent.critic = torch.nn.DataParallel(agent.critic)  # 如果有多个模型，也需要应用到critic
    print(f"Using device: {device}")
    
    for episode_i in range(NUM_EPISODE):
        state = env.reset()[0]
        episode_reward = 0
        done = False

        for step_i in range(NUM_STEP):
            action, value = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            done = True if (step_i + 1) == NUM_STEP else False
            agent.replay_buffer.add_memory(state, action, reward, value, done)
            state = next_state
            
            if (step_i + 1) % BATCH_SIZE == 0 or (step_i + 1) == NUM_STEP:
                agent.update()

        if episode_reward >= -100 and episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_policy()
            torch.save(agent.actor.state_dict(), model + f"actor_{timestamp}.pth")
            print(f"Best reward: {best_reward} at episode {episode_i}")
            
        REWARD_BUFFER[episode_i] = episode_reward
        print(f"Episode {episode_i} reward: {episode_reward}")
        
    
    

def register_quadrotor_env():
    gym.envs.registration.register(
        id='Quadrotor-v0',
        entry_point='__main__:QuadrotorEnv',
        max_episode_steps=10,
    )