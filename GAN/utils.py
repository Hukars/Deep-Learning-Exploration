# -*-coding:utf-8-*-
import torch
import torch.nn as nn

import numpy as np

using_gpu = True
device = torch.device('cuda' if using_gpu else 'cpu')
FloatTensor, LongTensor = (torch.FloatTensor, torch.LongTensor) if not using_gpu else (torch.cuda.FloatTensor,
                                                                                       torch.cuda.LongTensor)


def init_weight(m):
    if type(m) == nn.Linear:
        # size = m.weight.size()
        # size_in = size[1]
        size_in = m.in_features
        std = np.sqrt(1 / size_in)
        m.weight.data.normal_(0.0, std)
        m.bias.data.fill_(0.0)

