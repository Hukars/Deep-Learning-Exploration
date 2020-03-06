import torch
import torch.nn as nn


class PolicyDisc(nn.Module):
    def __init__(self, state_dim, action_num, hidden_size=(256, 256), activation='tanh'):
        super(PolicyDisc,self).__init__()
        self.is_disc_actions = True
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'relu':
            self.activation = torch.relu

        self.hidden_layer = nn.ModuleList()
        last_dim = state_dim
        for h in hidden_size:
            self.hidden_layer.append(nn.Linear(last_dim,h))
            last_dim = h

        self.action_head = nn.Linear(last_dim, action_num)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)

    def forward(self, x):
        """
        state->actions probs
        :param x: state
        :return:actions probs
        """
        for hidden in self.hidden_layer:
            x = self.activation(hidden(x))
        probs = torch.softmax(self.action_head(x), 0)

        return probs

    def select_action(self,x):
        """
        select a action by current policy
        :param x: state
        :return: action
        """
        probs = self.forward(x)
        print('probs',probs)
        return probs.multinomial(1)

    def log_prob(self, x, action):
        probs = self.forward(x)
        return torch.log(probs.gather(0, action.long()))