import trial01

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import gym

PATH = "test_nets/test_239.758.dat"
NUM_EPISODES = 10
device = "cpu"

net = trial01.Net(8, 64, 4)
net.load_state_dict(torch.load(PATH))

env = gym.make('LunarLander-v2')
for _ in range(NUM_EPISODES):
    state = env.reset()
    episode_reward = 0
    is_done = False
    while not is_done:
        env.render()
        state_a = np.array([state], copy=False)
        state_v = torch.FloatTensor(state_a).to(device)
        q_vals_v = net(state_v)
        _, act_v = torch.max(q_vals_v, dim=1)
        action = int(act_v.item())
        state, reward, is_done, _ = env.step(action)
        episode_reward += reward
        if is_done:
            state = env.reset()
            print("EPISODE REWARD: " + str(episode_reward))
env.close()
