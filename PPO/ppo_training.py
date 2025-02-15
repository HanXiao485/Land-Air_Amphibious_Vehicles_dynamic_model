import gym
import torch
import torch.nn as nn
import os
import time
import numpy as np

from ppo import Actor, Critic, ReplayMemory, PPOAgent

scenario = "Pendulum-v1"
env = gym.make(scenario, render_mode="human")
# env = gym.make(scenario)

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

for episode_i in range(NUM_EPISODE):
    state,others = env.reset()
    done = False
    episode_reward = 0

    for step_i in range(NUM_STEP):
        action, value = agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        done = True if (step_i + 1) == NUM_STEP else False
        agent.replay_buffer.add_memory(state, action, reward, value, done)
        state = next_state

        # env.render()

        if (step_i + 1) % BATCH_SIZE == 0 or (step_i + 1) == NUM_STEP:
            agent.update()
            
    if episode_reward >= -100 and episode_reward > best_reward:
        best_reward = episode_reward
        agent.save_policy()
        torch.save(agent.actor.state_dict(), model + f"actor_{timestamp}.pth")
        print(f"Best reward: {best_reward} at episode {episode_i}")

    REWARD_BUFFER[episode_i] = episode_reward
    print(f"Episode {episode_i} reward: {episode_reward}")

env.close()