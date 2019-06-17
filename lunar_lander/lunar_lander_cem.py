import gym
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def generate_batch(env, batch_size, t_max=5000):
    activation = nn.Softmax(dim=0)
    batch_actions, batch_states, batch_rewards = [], [], []

    for b in range(batch_size):
        states, actions = [], []
        total_reward = 0
        s = env.reset()
        for t in range(t_max):

            s_tensor = torch.Tensor(s)
            act_probs = activation(net(s_tensor))
            a = np.random.choice(len(act_probs), p=act_probs.detach().numpy())

            new_s, r, done, info = env.step(a)

            # record sessions like you did before
            states.append(s)
            actions.append(a)
            total_reward += r

            s = new_s
            if done:
                batch_actions.append(actions)
                batch_states.append(states)
                batch_rewards.append(total_reward)
                break

    return batch_states, batch_actions, batch_rewards


def filter_batch(states_batch, actions_batch, rewards_batch, percentile=50):
    reward_threshold = np.percentile(rewards_batch, percentile)

    elite_states = []
    elite_actions = []

    for i in range(len(rewards_batch)):
        if rewards_batch[i] > reward_threshold:
            for j in range(len(states_batch[i])):
                elite_states.append(states_batch[i][j])
                elite_actions.append(actions_batch[i][j])

    return elite_states, elite_actions


batch_size = 100
session_size = 500
percentile = 80
hidden_size = 200
completion_score = 200
learning_rate = 0.01

env = gym.make("LunarLander-v2")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# neural network
net = Net(n_states, hidden_size, n_actions)
# loss function
objective = nn.CrossEntropyLoss()
# optimisation function
optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)

for i in range(session_size):
    # generate new sessions
    batch_states, batch_actions, batch_rewards = generate_batch(env, batch_size, t_max=5000)

    elite_states, elite_actions = filter_batch(batch_states, batch_actions, batch_rewards, percentile)

    optimizer.zero_grad()
    action_scores_v = net(torch.Tensor(elite_states))
    loss_v = objective(action_scores_v, torch.LongTensor(elite_actions))
    loss_v.backward()
    optimizer.step()

    # show results
    mean_reward, threshold = np.mean(batch_rewards), np.percentile(batch_rewards, percentile)
    print("%d: loss=%.3f, reward_mean=%.1f, reward_threshold=%.1f" % (
        i, loss_v.item(), mean_reward, threshold))

    # check if
    if np.mean(batch_rewards) > completion_score:
        print("Environment has been successfully completed!")

# save the model
torch.save(net, 'model_best.pth.tar')

#record sessions
import gym.wrappers
env = gym.wrappers.Monitor(gym.make("LunarLander-v2"), directory="videos", force=True)
generate_batch(env, 1, t_max=5000)
env.close()

