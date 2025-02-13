import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from gymnasium import register
from rl_test.quadrotor_env import QuadrotorEnv  # 确保引入 QuadrotorEnv 类

from gym import envs
print(envs.registry.all())  # 输出中应包含'QuadRotor-v0'


if __name__ == "__main__":
    # 创建环境
    env = gym.make('Quadrotor-v0', mass=3.18, inertia=[0.029618, 0.069585, 0.042503], drag_coeffs=[0.0, 0.0], gravity=9.81, pid_params={
        'mass': 3.18,
        'gravity': 9.81,
        'desired_position': [0, 0, 5],
        'desired_velocity': [0, 0, 0],
        'desired_attitude': [0, 0, 0],
        'dt': 0.1
    })
    
    # 配置TensorBoard日志
    log_dir = "./tensorboard_logs/"

    # 配置和训练模型
    model = PPO("MlpPolicy", env, verbose=1,
                n_steps=256,
                batch_size=64,
                policy_kwargs=dict(
                    net_arch=dict(pi=[128, 128], vf=[128, 128]),  # 缩小网络规模
                    activation_fn=torch.nn.Tanh),  # 添加tanh激活函数
                learning_rate=1e-4,  # 降低学习率
                clip_range=0.2,  # 使用默认PPO裁剪范围
                max_grad_norm=0.5,  # 添加梯度裁剪
                device='cpu',
                tensorboard_log=log_dir)  # 将TensorBoard日志路径添加到模型配置中
    model.learn(total_timesteps=1e6)
    model.save("quadrotor_model")
