from typing import List
import random
import torch
import torch.nn as nn


class Transition:
    def __init__(self, state, action, next_state, reward, done):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.done = done


class ReplayMemory:

    def __init__(self, capacity: int, device):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.position_of_newest_sample = 0
        self.device = device

    def push(self, transition: Transition):
        """Saves an episode"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position_of_newest_sample = self.position
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def sample_cer(self, batch_size: int) -> List[Transition]:
        """
        DO NOT USE FOR NOW! DOESN'T WORK WELL!
        See : https://arxiv.org/pdf/1712.01275.pdf
        :param batch_size:
        :return:
        """
        batch = random.sample(self.memory, batch_size - 1)
        batch.append(self.memory[self.position_of_newest_sample])
        return batch

    def get_torch_sarsas_from_transitions(self, transitions: List[Transition]) -> (torch.FloatTensor, torch.LongTensor,
                                                                                   torch.FloatTensor, torch.FloatTensor,
                                                                                   torch.FloatTensor):
        states = torch.FloatTensor([transition.state for transition in transitions]).to(self.device)
        actions = torch.LongTensor([transition.action for transition in transitions]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([transition.next_state for transition in transitions]).to(self.device)
        rewards = torch.FloatTensor([transition.reward for transition in transitions]).unsqueeze(1).to(self.device)
        terminals = torch.FloatTensor([transition.done for transition in transitions]).unsqueeze(1).to(self.device)
        return states, actions, next_states, rewards, terminals

    def mean_qvalues(self, critic: nn.Module) -> torch.FloatTensor:
        """
        Given a critic model, this class calculates the Q-values on the entire Replay Buffer
        :param critic:
        :return:
        """
        all_samples = self.sample(len(self.memory))
        state_batch, _, _, _, _ = self.get_torch_sarsas_from_transitions(all_samples)
        qvalues = critic(state_batch)
        mean_q_values = torch.mean(qvalues, 0)
        return mean_q_values

    def __len__(self) -> int:
        return len(self.memory)
