import torch
import torch.nn as nn
import numpy as np
import math

using_gpu = True
device = torch.device('cuda' if using_gpu else 'cpu')


def to_device(current_device, *args):
    return [x.to(current_device) for x in args]


def init_weight(m):
    std = 0.0
    if type(m) == nn.Linear:
        # size = m.weight.size()
        # size_in = size[1]
        size_in = m.in_features
        std = np.sqrt(1 / size_in)
        m.weight.data.normal_(0.0, std)
        m.bias.data.fill_(0.0)
    elif type(m) == nn.Conv2d:
        std = m.in_channels / m.out_channels
        m.weight.data.normal_(0.0, std)
        m.bias.data.fill_(0.0)




def gaussian_density(variable, mean, std, log_std):
    var = std.pow(2)
    log_density = -(variable-mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


# hard update and soft update
def hard_update(target, source):
    target.load_state_dict(source.state_dict())


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

