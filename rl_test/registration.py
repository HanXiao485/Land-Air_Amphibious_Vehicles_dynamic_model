import gym
from gym import envs

# 获取所有已注册的环境
registered_envs = envs.registry.all()

# 打印所有环境的 ID
print([env.id for env in registered_envs])