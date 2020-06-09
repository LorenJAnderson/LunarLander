import gym
import numpy as np

import trial01

import torch
import torch.nn as nn
import torch.optim as optim

PATH = "test_nets/test_0.000.dat"
NUM_EPISODES = 10
DEVICE = "cpu"

net = trial01.Net(8, 64, 4)
net.load_state_dict(torch.load(PATH))

env = gym.make('LunarLander-v2')
for _ in range(NUM_EPISODES):
    state = env.reset()
    is_done = False
    while not is_done:
        env.render()
        state_a = np.array([state], copy=False)
        state_v = torch.FloatTensor(state_a).to(DEVICE)
        q_vals_v = net(state_v)
        _, act_v = torch.max(q_vals_v, dim=1)
        action = int(act_v.item())
        state, reward, is_done, _ = env.step(action)
        if is_done:
            state = env.reset()
env.close()
