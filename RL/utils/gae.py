import torch
from RL.utils.tools import to_device


def estimate_advantages(rewards, values, masks, gamma, lambda_, device):
    size = rewards.shape[0]
    advantages = torch.zeros(size, 1)

    pre_advantage = 0
    pre_value = 0
    for i in reversed(range(size)):
        shaped_reward = rewards[i, 0] + gamma * pre_value * masks[i, 0] - values[i, 0]
        advantages[i, 0] = shaped_reward + gamma * lambda_ * pre_advantage * masks[i, 0]

        pre_value = values[i, 0]
        pre_advantage = advantages[i, 0]

    returns = values + advantages
    #advantages = (advantages - advantages.mean()) / advantages.std()

    to_device(device, advantages, returns)
    return advantages, returns

