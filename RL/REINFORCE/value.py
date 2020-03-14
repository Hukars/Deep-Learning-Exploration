# -*-coding:utf-8-*-
import torch
import torch.nn as nn


class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128)):
        super().__init__()
        self.activation = torch.tanh
        last_dim = state_dim
        self.hidden_layer = nn.ModuleList()
        for h in hidden_size:
            self.hidden_layer.append(nn.Linear(last_dim, h))
            last_dim = h

        self.value_output = nn.Linear(last_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.1)

    def forward(self, x):
        for hidden in self.hidden_layer:
            x = self.activation(hidden(x))
        return self.value_output(x)
