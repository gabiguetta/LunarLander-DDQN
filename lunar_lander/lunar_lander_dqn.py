import copy

import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import time

from dqn_common.replay_memory import Transition, ReplayMemory

"""
Landing pad is always at coordinates (0,0). 
Coordinates are the first two numbers in state vector. 
Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. 
If lander moves away from landing pad it loses reward back. 
Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. 
Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. 
Landing outside landing pad is possible. 
Fuel is infinite, so an agent can learn to fly and then land on its first attempt. 
Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.
"""


class DeulingNet(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(DeulingNet, self).__init__()
        self.fc1 = nn.Linear(obs_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, n_actions)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        a = self.fc3(x)
        v = self.fc4(x)
        q = v + (a - torch.mean(a))
        return q


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

RENDER_INT = 1
PLOT_INT = 10
STOP_EPS = 0.001
START_EPS = 0.1
NUM_TRAIN_EPISODES = 100000
eps_grid = np.linspace(START_EPS, STOP_EPS, NUM_TRAIN_EPISODES)

GAMMA = 0.99
LR = 0.001
MEM_SIZE = 1000000
MAX_EPISODE_LENGTH = 1000
NUM_TEST_EPISODES = 10
BATCH_SIZE = 32

TARGET_NET_UPDATE_INT = 10000
is_double_dqn_applied = True  # see: http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12389/11847

if is_double_dqn_applied:
    assert TARGET_NET_UPDATE_INT > 1, "Double DQN is relevant only when TARGET_NET_UPDATE_INT>1"

memory = ReplayMemory(MEM_SIZE, device)

env_name = "LunarLander-v2"  # "LunarLander-v2" # 'CartPole-v0' # 'Pendulum-v0' # 'MountainCar-v0' # 'Pong-v0'
env = gym.make(env_name)

n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

model = DeulingNet(n_states, n_actions).to(device)


def random_action() -> int:
    return random.randrange(n_actions)


def optimize(optimizer, criterion, target_model):

    optimizer.zero_grad()

    transitions = memory.sample(min(BATCH_SIZE, len(memory)))

    state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = \
        memory.get_torch_sarsas_from_transitions(transitions)

    Q_s_chosen_action = model(state_batch).gather(1, action_batch)

    model_for_target_action_selection = model if is_double_dqn_applied else target_model
    Q_next_s_max = target_model(next_state_batch).gather(1, torch.argmax(
        model_for_target_action_selection(next_state_batch), 1).unsqueeze(1))

    target_Q_val = reward_batch + (GAMMA * Q_next_s_max * (1 - terminal_batch))
    loss = criterion(Q_s_chosen_action, target_Q_val.detach())
    loss.backward()
    optimizer.step()

    return loss.item()


def train():

    writer = SummaryWriter()

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    run_loss = []
    run_rewards = []
    target_model = copy.deepcopy(model)
    j = 0
    for episode_num in range(NUM_TRAIN_EPISODES):
        # start a new episode
        state = env.reset()
        done = False
        eps = eps_grid[episode_num]
        episode_loss = []
        episode_step = 0
        episode_reward = 0
        while not done and episode_step < MAX_EPISODE_LENGTH:
            if episode_num % RENDER_INT == 0:
                img = env.render(mode='rgb_array')
            Q_s = model(torch.Tensor(state).to(device))
            action = torch.argmax(Q_s).item() if random.random() > eps else random_action()
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            memory.push(Transition(state, action, next_state, reward, done))
            episode_loss.append(optimize(optimizer, criterion, target_model))
            if j % TARGET_NET_UPDATE_INT == 0:
                target_model = copy.deepcopy(model)
            state = next_state
            episode_step += 1
            j += 1
        run_loss.append(np.mean(episode_loss))
        run_rewards.append(episode_reward)
        if episode_num % PLOT_INT == 0:
            mem_qvalues = memory.mean_qvalues(critic=model)
            print('iter %d (eps=%f) : accumulated reward is: %f, loss is: %f' % (
                episode_num, eps, np.mean(run_rewards), np.mean(run_loss)))
            writer.add_scalar("mean episodic reward", np.mean(run_rewards), episode_num)
            writer.add_scalar("loss", np.mean(run_loss), episode_num)
            writer.add_scalar("avg_max_Q_val_on_mem", torch.max(mem_qvalues).item(), episode_num)
            run_loss = []
            run_rewards = []
        # save the model
        torch.save(model, 'latest_lunar_model.tar')


def test():
    model = torch.load('latest_lunar_model.tar')
    for i in range(NUM_TEST_EPISODES):
        state = env.reset()
        done = False
        episode_reward = 0
        ep_step = 0
        while not done and ep_step < MAX_EPISODE_LENGTH:
            env.render()
            Q_s = model(torch.Tensor(state))
            action = torch.argmax(Q_s).item()
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            ep_step += 1
        print('Reward received for test scenario #%d is : %f' % (i, episode_reward))
        time.sleep(1)


if __name__ == "__main__":
    train()
    # test()
