# -*-coding:utf-8-*-
import gym
import itertools
import argparse
import time
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from RL.DQN.wrappers import make_env, LazyFrames
from RL.models.critic import ConvolutionalActionCritic
from RL.utils import *

# 环境和运算设备
env = gym.make('Breakout-v0')
env = make_env(env)

# 参数
parser = argparse.ArgumentParser()
parser.add_argument('--using_double', type=bool, default=False, help='if using_double, use Double DQN')
parser.add_argument('--max_steps', type=int, default=10000000, help='number of epochs of training')
parser.add_argument('--memory_capacity', type=int, default=1000000, help='the replay memory maximum size')
parser.add_argument('--batch_size', type=int, default=64, help='the batch size for policy improvement')
parser.add_argument('--lr', type=float, default=1e-4, help='the learning rate of gradient descent')
parser.add_argument('--epsilon_start', type=float, default=1.0, help='the epsilon value at the beginning')
parser.add_argument('--epsilon_decay', type=int, default=1000000, help='epsilon anneal over this times')
parser.add_argument('--epsilon_end', type=float, default=0.1, help='the final epsilon value after annealing')
parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
parser.add_argument('--target_update', type=int, default=1000, help='the period for update the target network')
parser.add_argument('--states_number', type=int, default=40000, help='sample some states as the measure set')
parser.add_argument('--render', type=bool, default=True, help='if render the picture ')
parser.add_argument('--seed', type=int, default=1, help='random seed')
args = parser.parse_args()


# Initialize some variable
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# state_dim = env.observation_space.shape[0]
action_nums = int(env.action_space.n)
is_disc_action = len(env.action_space.shape) == 0
action_dim = 1 if is_disc_action else env.action_space.shape[0]

print(action_nums)
policy_net = ConvolutionalActionCritic(action_nums).to(device)
target_net = ConvolutionalActionCritic(action_nums).to(device)
target_net.load_state_dict(policy_net.state_dict())
memory = Memory(args.memory_capacity)

optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
func = nn.MSELoss()
# writer = SummaryWriter()


def select_action(current_state, action_nums):
    global total_steps
    current_epsilon = (args.epsilon_end + (args.epsilon_start-args.epsilon_end)*(1 - total_steps/args.epsilon_decay)
                       if total_steps<=args.epsilon_decay else args.epsilon )
    uniform_number = random.uniform(0, 1)
    if uniform_number < current_epsilon:
        return random.randint(0, action_nums-1)
    else:
        current_state = torch.tensor(current_state).unsqueeze(0).to(device)
        with torch.no_grad():
            return policy_net(current_state.float()).max(1)[1].cpu().numpy()[0]


def optimize_model():
    global update
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
    if not args.using_double:
        with torch.no_grad():
            target_values = rewards + args.gamma * masks * target_net(next_states).max(1, keepdim=True)[0]
        loss = func(estimated_values, target_values)
    else:
        with torch.no_grad():
            target_values = rewards + args.gamma * masks * torch.gather(target_net(next_states), 1,
                                                                        policy_net(next_states).max(1)[1].unsqueeze(1))
        loss = func(estimated_values, target_values)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    if update % args.target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())


def observation_to_state(obs: LazyFrames):
    """
    84*84*4 to 4*84*84
    :param obs:
    :return:
    """
    obs = np.array(obs)
    return obs.transpose((2, 0, 1))


# Training the model
total_steps = 0
update = 0
max_episode_reward = 0
for i_episode in itertools.count(1):
    print(f'episode {i_episode}, total_step:{total_steps}')
    start = time.time()
    observation = env.reset()
    state = observation_to_state(observation)
    done = False
    episode_reward = 0

    while not done:
        if args.render:
            env.render()
        action = select_action(state, action_nums)
        observation, reward, done, _ = env.step(action)
        next_state = observation_to_state(observation)
        memory.complex_push(state, action, reward, next_state, 0 if done else 1, 0)
        episode_reward += reward
        state = next_state
        total_steps += 1
        update += 1
        optimize_model()
    time_cost = time.time() - start
    max_episode_reward = max(episode_reward, max_episode_reward)
    print(f'Complete episode{i_episode}, time_cost:{time_cost}, episode_reward:{episode_reward},'
          f' max_episode_reward:{max_episode_reward}')
    if total_steps > args.max_steps:
        break

torch.save(policy_net.state_dict(), 'q_net.pt')
print('Training Complete!')
# writer.close()
env.close()

