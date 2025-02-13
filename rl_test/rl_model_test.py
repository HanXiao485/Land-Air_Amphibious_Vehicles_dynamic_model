import gymnasium as gym
from stable_baselines3 import PPO

# 假设你已经有一个训练好的模型，存放在指定路径
model_path = "./trained_model.zip"  # 训练好的模型文件路径

# 创建环境
env = gym.make('Quadrotor-v0')  # 假设环境为 Quadrotor-v0，替换为实际环境ID

# 载入已经训练好的模型
model = PPO.load(model_path)

# 在环境中进行验证（测试）
obs, _ = env.reset()  # 获取环境的初始状态
done = False
total_reward = 0

# 进行验证
while not done:
    # 使用模型预测动作
    action, _ = model.predict(obs, deterministic=True)
    
    # 执行动作并获取新的状态和奖励
    obs, reward, done, _, _ = env.step(action)
    
    # 累加奖励
    total_reward += reward

    # 渲染环境状态
    env.render()

# 打印最终的奖励
print(f"Total reward: {total_reward}")
env.close()  # 关闭环境
