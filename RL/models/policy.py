# -*-coding:utf-8-*-
from RL.utils import *
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class DiscretePolicy(nn.Module):
    """
    Policy for some simple environments with discrete action space
    """
    def __init__(self, state_dim, action_num, hidden_size=256):
        super(DiscretePolicy, self).__init__()
        self.is_disc_actions = True

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_num),
            nn.Softmax()
        )

        self.model.apply(init_weight)

    def forward(self, x):
        probs = self.model(x)

        return probs

    def select_action(self, x):
        """
        select a action by current policy
        :param x: state
        :return: action
        """
        probs = self.forward(x)
        print('probs', probs)
        return probs.multinomial(1)

    def log_prob(self, x, action):
        probs = self.forward(x)
        return torch.log(probs.gather(1, action.long()))


class GaussianPolicy(nn.Module):
    """
    For continuous action-policy, we use a Gaussian distribution to model it, for PPO actor, TRPO actor, GAIL actor
    """
    def __init__(self, state_dim, action_dim, hidden_size=256, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, action_dim),
        )

        self.is_disc_actions = False

        # action rescaling
        if action_space is not None:
            self.action_scale = torch.tensor(1.).to(device)
            self.action_bias = torch.tensor(0.).to(device)
        else:
            self.action_scale = torch.tensor((action_space.high - action_space.low) / 2.).float().to(device)
            self.action_bias = torch.tensor((action_space.high + action_space.low) / 2.).float().to(device)

        self.action_std = nn.Parameter(torch.ones(1, action_dim))
        self.model.apply(init_weight)

    def forward(self, x):
        action_mean = self.model(x)
        action_std = self.action_std.expand_as(action_mean)
        return action_mean, action_std

    def select_action(self, x):
        action_mean, action_std = self.forward(x)
        normal = Normal(action_mean, action_std)
        return torch.tanh(normal.sample()) # * self.action_scale + self.action_bias

    def log_prob(self, x, y):
        action_mean, action_std = self.forward(x)
        return gaussian_density(y, action_mean, action_std, torch.log(action_std))


class ReparameterGaussianPolicy(GaussianPolicy):
    def __init__(self, state_dim, action_dim, hidden_size=256, action_space=None):
        super(ReparameterGaussianPolicy, self).__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            action_space=action_space
        )

    def rsample(self, state):
        mean, std = super(ReparameterGaussianPolicy, self).forward(state)
        std = torch.clamp(std, min=math.exp(LOG_SIG_MIN), max=math.exp(LOG_SIG_MAX))
        normal = Normal(mean, std)
        # reparameterization trick (mean + std * N(0,1))
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == '__main__':
    policy = DiscretePolicy(4, 3)
    a = torch.tensor([[1.1, 2, 3, 4], [2.3, 4, 5, 6]])
    action = torch.tensor([[1.0], [2]])
    x = policy(a)
    print(x.gather(1, action.long()))