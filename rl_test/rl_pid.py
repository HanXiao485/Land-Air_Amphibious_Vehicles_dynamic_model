import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

class DoubleCascadePIDEnv(gym.Env):
    """
    双环串级PID控制环境
    外环：位置控制
    内环：速度控制
    """
    def __init__(self):
        super(DoubleCascadePIDEnv, self).__init__()
        
        # 系统参数
        self.dt = 0.01  # 仿真时间步长
        self.max_steps = 200  # 最大步数
        self.target = 10.0  # 目标位置
        self.velocity_target = 0  # 初始化 velocity_target 为 0 
        
        # 动作空间：6个PID参数 [外环Kp, Ki, Kd, 内环Kp, Ki, Kd]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([2, 2, 2, 2, 2, 2], dtype=np.float32),
            dtype=np.float32
        )
        
        # 状态空间：外环误差，外环积分，外环微分，内环误差，内环积分，内环微分
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )
        
        # 系统模型参数（二阶系统示例）
        self.mass = 1.0     # 质量
        self.damping = 0.1  # 阻尼
        
        # 重置环境
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # 确保 Gym 兼容性
        # 初始状态
        self.position = 0.0  # 位置
        self.velocity = 0.0  # 速度
        self.step_count = 0
        
        # 外环PID记忆
        self.outer_integral = 0.0
        self.outer_last_error = 0.0
        
        # 内环PID记忆
        self.inner_integral = 0.0
        self.inner_last_error = 0.0
        
        obs = self._get_obs()
        info = {}  # 额外的环境信息，Gym 需要返回
        
        return obs, info  # 这里必须返回元组

    def _get_obs(self):
        """获取观测值"""
        outer_error = self.target - self.position
        inner_error = self.velocity_target - self.velocity
        return np.array([ 
            outer_error,
            self.outer_integral,
            outer_error - self.outer_last_error,
            inner_error,
            self.inner_integral,
            inner_error - self.inner_last_error
        ], dtype=np.float32)

    def step(self, action):
        # 解析动作参数
        outer_kp, outer_ki, outer_kd, inner_kp, inner_ki, inner_kd = action
        
        # 外环PID控制（位置环）
        outer_error = self.target - self.position
        self.outer_integral += outer_error * self.dt
        outer_derivative = (outer_error - self.outer_last_error) / self.dt
        
        # 外环输出作为内环目标速度
        self.velocity_target = (
            outer_kp * outer_error +
            outer_ki * self.outer_integral +
            outer_kd * outer_derivative
        )
        
        # 内环PID控制（速度环）
        inner_error = self.velocity_target - self.velocity
        self.inner_integral += inner_error * self.dt
        inner_derivative = (inner_error - self.inner_last_error) / self.dt
        
        # 计算控制力
        force = (
            inner_kp * inner_error +
            inner_ki * self.inner_integral +
            inner_kd * inner_derivative
        )
        
        # 系统动力学模型
        acceleration = (force - self.damping * self.velocity) / self.mass
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt
        
        # 保存误差用于微分计算
        self.outer_last_error = outer_error
        self.inner_last_error = inner_error
        
        # 计算奖励
        reward = -(
            0.5 * abs(outer_error) +  # 位置误差
            0.3 * abs(inner_error) +  # 速度误差
            0.1 * abs(force)          # 控制量约束
        )
        
        reward = float(reward)
        
        # 终止条件
        self.step_count += 1
        done = self.step_count >= self.max_steps  # 终止
        truncated = False  # Gym API 需要返回 `truncated`，在此设为 False
        
        return self._get_obs(), reward, done, truncated, {}

# 创建环境
env = DoubleCascadePIDEnv()
check_env(env)  # 检查环境兼容性

# 创建PPO模型
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 训练模型并记录奖励和损失
reward_list = []  # 存储奖励
loss_list = []    # 存储损失
callback = model.learn(total_timesteps=100000)

# 绘制奖励和损失曲线
def plot_reward_loss(reward_list, loss_list):
    plt.figure(figsize=(12, 6))
    
    # 绘制奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(reward_list, label="Reward")
    plt.title("Reward Curve")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend()
    
    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(loss_list, label="Loss", color="red")
    plt.title("Loss Curve")
    plt.xlabel("Timesteps")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 保存模型
model.save("dual_pid_ppo")

# 测试训练结果
obs, _ = env.reset()  # Extract only the observation part
for _ in range(200):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    reward_list.append(reward)
    loss_list.append(0)  # 你可以在此记录模型的损失（如果你有实现损失）
    # print(f"Position: {env.position:.2f}, Velocity: {env.velocity:.2f}")
    if done:
        break

# 绘制奖励和损失图
plot_reward_loss(reward_list, loss_list)
