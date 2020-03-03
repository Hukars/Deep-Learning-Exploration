import gym
from collections import namedtuple
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 环境和运算设备
env = gym.make('MountainCar-v0').unwrapped
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 参数
parser = argparse.ArgumentParser()
parser.add_argument('--n_epoch', type=int, default=500, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='the batch size for policy improvement')
parser.add_argument('--lr', type=float, default=1e-5, help='the learning rate of gradient descent')
parser.add_argument('--epsilon', type=float, default=0.2, help='the probability for choose a random action')
parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
parser.add_argument('--target_update', type=int, default=5, help='the period for update the target network')
parser.add_argument('--states_number', type=int, default=5000, help='sample some states as the measure set')
args = parser.parse_args()

# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        self.memory[self.position] = transition
        self.position = (self.position+1) % self.capacity

    def sample(self, batch_size):
        random_batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*random_batch))

    def __len__(self):
        return len(self.memory)


# action-value network
class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.model(x)


def random_states():
    """
    Use a random policy to sample some states as the measure set
    :return:state list
    """
    state_list = []
    env.reset()
    for i in range(args.states_number):
        observation, _, _, _ = env.step(env.action_space.sample())
        state_list.append(observation)
    return state_list


# Initialize some variable
env.reset()
state_dim = env.observation_space.shape[0]
action_nums = env.action_space.n
is_disc_action = len(env.action_space.shape) == 0
action_dim = 1 if is_disc_action else env.action_space.shape[0]

policy_net = Network(state_dim, action_nums).to(device)
target_net = Network(state_dim, action_nums).to(device)
target_net.load_state_dict(policy_net.state_dict())
memory = ReplayMemory(1000)

optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
func = nn.MSELoss()
writer = SummaryWriter()
common_states = random_states()


def select_action(current_state, action_nums):
    uniform_number = random.uniform(0, 1)
    if uniform_number < args.epsilon:
        action = random.randint(0, action_nums-1)
    else:
        current_state = torch.tensor(current_state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy_net(current_state.float()).max(1)[1].cpu().numpy()[0]
    return action


def optimize_model():
    # If  Replay Memory is not full, continue to sample
    if memory.__len__() < memory.capacity:
        return
    batch = memory.sample(args.batch_size)
    states = torch.tensor(batch.state).float().to(device)
    actions = torch.tensor(batch.action).unsqueeze(1).to(device)
    reward = torch.tensor(batch.reward).unsqueeze(1).to(device)
    next_state = torch.tensor(batch.next_state).float().to(device)

    estimated_values = torch.gather(policy_net(states), 1, actions)
    max_actions = torch.argmax(policy_net(next_state), 1).unsqueeze(1)
    target_values = reward + args.gamma * torch.gather(target_net(next_state), 1, max_actions)

    loss = func(estimated_values, target_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Training the model
for epoch in range(args.n_epoch):
    print(f'Epoch {epoch+1}')
    state = env.reset()
    print(f'state:{state}')
    for i in range(5000):
        env.render()
        action = select_action(state, action_nums)
        observation, reward, done, _ = env.step(action)
        memory.push(Transition(state, action, reward, observation))

        state = observation

        optimize_model()

        if done:
            print(f'After {i} step, we got the mountain top!')
            break

    if (epoch+1) % args.target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # caculate the average Q-value over common_states
    states = torch.tensor(common_states).float()
    max_action_values = policy_net(states).max(1)[0].sum()
    avg_action_value = max_action_values/args.states_number
    writer.add_scalars('Average Q on MountainCar-v0',{
        'average action value': avg_action_value
    }, epoch+1)

print('Training Complete!')
env.close()

