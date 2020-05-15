import random
import torch
from collections import namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'mask', 'old_log_prob'))
# Complex_Transition = namedtuple('Complex_Transition', (
#    'onehot_state', 'multihot_state', 'continuous_state', 'onehot_action', 'multihot_action', 'continuous_action',
#    'reward', 'next_onehot_state', 'next_multihot_state', 'next_continuous_state', 'mask', 'old_log_prob'))


class Memory:
    def __init__(self, capacity=10e6):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def complex_push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[int(self.position)] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            sample = random.sample(self.memory, batch_size)
            return Transition(*zip(*sample))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)

    def clear_memory(self):
        del self.memory[:]


if __name__ == "__main__":
    memory = Memory()
    for i in range(4):
        state = [[i + 1, 2, 3], [i + 1, 5, 6]]
        action = [[0], [1]]
        reward = [[0.4], [0.6]]
        next_state = [[i + 2, 3, 3], [i + 2, 6, 6]]
        if i+1 <= 3:
            mask = torch.ones(2, 1)
        else:
            mask = torch.zeros(2, 1)
        log_prob = [[-0.12], [-0.14]]
        memory.push(state, action, reward, next_state, mask, log_prob)
    a = memory.sample().mask
    print(a)
    print(torch.cat(a, 1))
    print(torch.cat(a, 1).reshape(8, -1))

