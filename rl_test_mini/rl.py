import torch
import gym
from gym import spaces
import numpy as np
import math
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv

class DroneTakeoffEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(DroneTakeoffEnv, self).__init__()
        
        # 如果需要渲染模式参数
        self.render_mode = render_mode

        # 无人机的状态空间：高度和速度
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([10, 10]), dtype=np.float32)

        # PID参数空间：[Kp, Ki, Kd]，取值范围假设是 [0, 10] 之内
        self.action_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([10, 10, 10]), dtype=np.float32)
        
        # 初始状态
        self.state = np.array([0.0, 0.0])  # 高度为0，速度为0
        self.target_altitude = 10  # 目标高度为10米
        
        # PID控制器的初始化
        self.prev_error = 0
        self.integral = 0

    def reset(self):
        # 重置环境状态
        self.state = np.array([0.0, 0.0])
        self.prev_error = 0
        self.integral = 0
        return self.state

    def step(self, action):
        # 提取PID参数
        Kp, Ki, Kd = action
        print(f"PID parameters: Kp={Kp}, Ki={Ki}, Kd={Kd}")

        # 计算误差
        error = self.target_altitude - self.state[0]  # 当前高度与目标高度的误差
        
        # PID控制算法
        self.integral += error
        derivative = error - self.prev_error
        control_output = Kp * error + Ki * self.integral + Kd * derivative
        
        # 更新速度和高度
        self.state[1] = control_output  # 将控制输出作为速度
        self.state[0] += self.state[1]  # 高度 = 速度 * 时间（假设时间步长为1）

        # 更新上一个误差
        self.prev_error = error

        # 设置奖励函数：高度接近目标高度时，奖励更高
        reward = -abs(self.target_altitude - self.state[0])  # 距离目标越近，奖励越高
        print(f"Current altitude: {self.state[0]} m, \n Current speed: {self.state[1]} m/s, \n Reward: {reward}")

        # 判断是否达到结束条件
        done = False
        if self.state[0] >= self.target_altitude or self.state[0] < 0:
            done = True

        return self.state, reward, done, {}

    def render(self):
        if self.render_mode == "human":
            print(f"Current altitude: {self.state[0]} m, Current speed: {self.state[1]} m/s")
        # 其他渲染模式的逻辑可以根据需要添加




# 配置TensorBoard日志
log_dir = "./tensorboard_logs/"

# 创建环境
env = DummyVecEnv([lambda: DroneTakeoffEnv(render_mode="human")])

# 初始化PPO模型
model = sb3.PPO("MlpPolicy", env, verbose=1,
                n_steps=512,
                batch_size=128,
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 256], vf=[256, 256]),  # 缩小网络规模
                    activation_fn=torch.nn.ReLU),  # 添加tanh激活函数
                learning_rate=3e-4,  # 降低学习率
                clip_range=0.3,  # 使用默认PPO裁剪范围
                max_grad_norm=0.7,  # 添加梯度裁剪
                device='cpu',
                tensorboard_log=log_dir)  # 将TensorBoard日志路径添加到模型配置中

# 开始训练
model.learn(total_timesteps=20000)

# 保存训练好的模型
model.save("drone_pid_model")

# 测试训练后的模型
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    # env.render()
    if done:
        break
