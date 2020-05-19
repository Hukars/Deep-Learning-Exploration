import gym
import time
import argparse
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from RL.models.agent import Agent
from RL.models.critic import ValueCritic
from RL.models.policy import GaussianPolicy
from RL.utils import *

"""
 parameters
"""
parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default="Hopper-v2", help="name of the environment to run")
parser.add_argument('--epoch', type=int, default=500, help="the total epoch for training")
parser.add_argument('--batch_size', type=int, default=2048, help="the approximate batch size")
parser.add_argument('--gamma', type=float, default=0.99, help="the discount factor")
parser.add_argument('--tau', type=float, default=0.95, help="gae factor")
parser.add_argument('--learning_rate', type=float, default=3e-4, help="the learning rate")
parser.add_argument('--l2_reg', type=float, default=1e-3, help="l2 regularization regression")
parser.add_argument('--seed', type=int, default=0, help="random seed")
args = parser.parse_args()

"""
 environment
"""
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

"""
 model
"""
policy = GaussianPolicy(state_dim, action_dim, 256, env.action_space).to(device)
critic = ValueCritic(state_dim).to(device)
running_state = ZFilter((state_dim,), clip=5)

"""
 optimizer
"""
optimizer_policy = optim.Adam(policy.parameters(), lr=args.learning_rate)
optimizer_critic = optim.Adam(critic.parameters(), lr=args.learning_rate)
loss_func = nn.MSELoss()

"""
 others
"""
agent = Agent(env, policy, running_state)
env.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
writer = SummaryWriter('VPG-Hopper-v2')


def vpg_step(states, advantages, returns, old_log_prob, num_trajectory):
    # update critic
    value_pred = critic(states)
    optimizer_critic.zero_grad()
    v_loss = loss_func(value_pred, returns)
    for parm in critic.parameters():
        v_loss += parm.pow(2).sum() * args.l2_reg
    v_loss.backward()
    optimizer_critic.step()
    # update policy
    assert advantages.shape == old_log_prob.shape
    p_loss = -((old_log_prob * advantages).sum()) / num_trajectory
    optimizer_policy.zero_grad()
    p_loss.backward()
    optimizer_policy.step()


def main_loop():
    for e in range(args.epoch):
        start = time.time()
        print(f"Start epoch{e+1}")
        print(f"Begin sampling")
        batch, log = agent.process_sample(args.batch_size)
        print(f"Sampling complete")
        num_trajs = log['num_episodes']
        states = torch.cat(batch.state, 0).to(device)
        # actions = torch.cat(batch.action, 0).to(device)
        rewards = torch.cat(batch.reward, 0).unsqueeze(1).to(device)
        # next_states = torch.cat(batch.next_state, 0).to(device)
        masks = torch.cat(batch.mask, 0).unsqueeze(1).to(device)
        old_log_prob = torch.cat(batch.old_log_prob, 0).to(device)
        with torch.no_grad():
            values = critic(states)

        advantages, returns = estimate_advantages(rewards, values, masks, args.gamma, args.tau)

        vpg_step(states, advantages, returns, old_log_prob, num_trajs)
        time_cost = time.time() - start

        print(f"End Epoch{e+1}, time_cost:{time_cost}, total_reward:{log['total_reward']}, average_reward:{log['avg_reward']}, "
              f"maximum_reward:{log['max_reward']}")
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main_loop()
