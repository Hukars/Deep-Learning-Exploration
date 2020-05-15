from RL.utils import *


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Discriminator, self).__init__()

        self.discriminator_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        self.discriminator_model.apply(init_weight)

    def forward(self, state, action):
        state_action = torch.cat((state, action), dim=1)
        x = self.discriminator_model(state_action)
        return x
