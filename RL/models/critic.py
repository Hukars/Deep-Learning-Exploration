from RL.utils import *

"""
Input are the environment state
"""


class ValueCritic(nn.Module):
    def __init__(self, state_dim, hidden_size=256):
        super(ValueCritic, self).__init__()

        self.value_model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.value_model.apply(init_weight)

    def forward(self, state):
        return self.value_model(state)


class ActionCritic(nn.Module):
    def __init__(self, state_dim, action_dim, as_policy=False, hidden_size=256):
        super(ActionCritic, self).__init__()
        if not as_policy:
            self.action_value_model = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, 1)
            )
        else:
            self.action_value_model = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, action_dim)
            )

        self.action_value_model.apply(init_weight)

    def forward(self, input):
        return self.action_value_model(input)


"""
Input are high-dimensional raw data such as image 
"""


class ConvolutionalActionCritic(nn.Module):
    def __init__(self, n_actions):
        super(ConvolutionalActionCritic, self).__init__()

        self.convolution_module = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        self.full_connected_module = nn.Sequential(
            nn.Linear(9 * 9 * 32, 256),
            nn.LeakyReLU(),
            nn.Linear(256, n_actions)
        )

        self.convolution_module.apply(init_weight)
        self.full_connected_module.apply(init_weight)

    def forward(self, x):
        x = self.convolution_module(x)
        return self.full_connected_module(x.view(x.size(0), -1))


if __name__ == '__main__':
    q_net = ActionCritic(4, 3, True)
    a = torch.tensor([[1.1, 2, 3, 4], [2, 3, 4, 5.1]])
    print(q_net(a).max(1)[1].unsqueeze(1))
