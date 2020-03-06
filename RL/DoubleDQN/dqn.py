import gym
from collections import namedtuple
from tqdm import tqdm
import random
import argparse
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# 环境和运算设备
env = gym.make('MountainCar-v0').unwrapped
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 参数
parser = argparse.ArgumentParser()
parser.add_argument('--n_epoch', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='the batch size for policy improvement')
parser.add_argument('--lr', type=float, default=1e-2, help='the learning rate of gradient descent')
parser.add_argument('--epsilon', type=float, default=0.1, help='the probability for choose a random action')
parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
parser.add_argument('--target_update', type=int, default=100, help='the period for update the target network')
parser.add_argument('--states_number', type=int, default=40000, help='sample some states as the measure set')
parser.add_argument('--seed', type=int, default=1, help='random seed')
args = parser.parse_args()

# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'mask'))


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
            nn.Linear(input_size, 64),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Linear(64, output_size)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)

    def forward(self, x):
        return self.model(x)


def random_states():
    """
    Use a random policy to sample some states as the measure set
    :return:state list
    """
    state_list = []
    for i in range(int(args.states_number/2000)):
        env.reset()
        for j in range(2000):
            observation, _, _, _ = env.step(env.action_space.sample())
            state_list.append(observation)
    return state_list


# Initialize some variable
env.seed(args.seed)
torch.manual_seed(args.seed)

state_dim = env.observation_space.shape[0]
action_nums = env.action_space.n
is_disc_action = len(env.action_space.shape) == 0
action_dim = 1 if is_disc_action else env.action_space.shape[0]

policy_net = Network(state_dim, action_nums).to(device)
target_net = Network(state_dim, action_nums).to(device)
target_net.load_state_dict(policy_net.state_dict())
memory = ReplayMemory(5000)

optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
func = nn.MSELoss()
writer = SummaryWriter()



def select_action(current_state, action_nums):
    uniform_number = random.uniform(0, 1)
    if uniform_number < args.epsilon:
        return random.randint(0, action_nums-1)
    else:
        current_state = torch.tensor(current_state).unsqueeze(0).to(device)
        with torch.no_grad():
            return policy_net(current_state.float()).max(1)[1].cpu().numpy()[0]


def optimize_model():
    # If  Replay Memory is not full, continue to sample
    if memory.__len__() < args.batch_size:
        return
    batch = memory.sample(args.batch_size)
    states = torch.tensor(batch.state).float().to(device)
    actions = torch.tensor(batch.action).unsqueeze(1).to(device)
    rewards = torch.tensor(batch.reward).unsqueeze(1).to(device)
    next_states = torch.tensor(batch.next_state).float().to(device)
    masks = torch.tensor(batch.mask).unsqueeze(1).float().to(device)

    estimated_values = torch.gather(policy_net(states), 1, actions)
    with torch.no_grad():
        target_values = rewards + args.gamma * masks * policy_net(next_states).max(1, keepdim=True)[0]

    loss = func(estimated_values, target_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Training the model
for epoch in range(args.n_epoch):
    print(f'Epoch {epoch+1}')
    state = env.reset()
    episodes_reward = []
    step = 0
    total_reward = 0
    for i in tqdm(range(3000)):
        env.render()
        action = select_action(state, action_nums)
        observation, reward, done, _ = env.step(action)
        memory.push(Transition(state, action, reward, observation, 0 if done else 1))

        state = observation
        total_reward += math.pow(args.gamma, step) * reward
        step += 1

        optimize_model()

        if done:
            episodes_reward.append(total_reward)
            total_reward = 0
            step = 0
            state = env.reset()
    if len(episodes_reward) == 0:
        episodes_reward.append(total_reward)

    avg_reward = torch.tensor(episodes_reward).float().mean()
    print(f'epoch:{epoch+1},avg_reward:{avg_reward}')
    writer.add_scalars('Average rewards per episode',{
        'average reward per episode': avg_reward
    }, epoch+1)

print('Training Complete!')
writer.close()
env.close()

