# -*-coding:utf-8-*-
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import gym
import argparse
import math
from collections import namedtuple

from RL.REINFORCE.agent import Experience
from RL.models.policy_disc_action import PolicyDisc


def get_args():
    parser = argparse.ArgumentParser(description="REINFORCE Algorithm")
    parser.add_argument('--alpha', type=float, default=3e-2, metavar='G1', help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G2', help='discount factor')
    parser.add_argument('--env-name', default='MountainCar-v0', metavar='G3', help='environment')
    parser.add_argument('--epoch', default=500, metavar='N1', help='iteration times')
    parser.add_argument('--seed', type=int, default=1, metavar='N2', help='random seed (default: 1)')
    return parser


def optimize_policy(total_steps):
    for i in range(total_steps):
        state, action, Gt = experience.get_item(i)
        loss = math.pow(args.gamma, i) * Gt * policy_net.log_prob(state, action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    # arguments
    args = get_args().parse_args()
    # computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # transition tuple
    transition = namedtuple('transition', ('state', 'action', 'reward', 'next_state'))

    dtype = torch.float64
    torch.set_default_dtype(dtype)

    env = gym.make(args.env_name).unwrapped
    state_dim = env.observation_space.shape[0]
    is_disc_action = len(env.action_space.shape) == 0
    if True:
        action_num = env.action_space.n
        policy_net = PolicyDisc(state_dim, action_num).to(device)

    experience = Experience(args.gamma)
    optimizer = optim.Adam(policy_net.parameters(), lr=args.alpha)

    # seeding
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    writer = SummaryWriter()

    for e in range(args.epoch):
        experience.reset()
        state = torch.from_numpy(env.reset()).to(dtype).to(device)
        print(f"epoch {e} starts")
        episode_reward = []
        total_reward = 0
        T = 0
        for i in range(8000):
            env.render()
            with torch.no_grad():
                action = policy_net.select_action(state)
            next_state, reward, done, _ = env.step(action.item())

            current_transition = transition(state, action, reward, next_state)
            experience.transition_list.append(current_transition)
            total_reward += math.pow(args.gamma, T) * reward
            T += 1
            state = torch.from_numpy(next_state).to(dtype).to(device)

            if done:
                episode_reward.append(total_reward)
                experience.calculate_return()
                optimize_policy(T)
                total_reward = 0
                T = 0

        # experience.calculate_return()
        # optimize_policy(T)
        """clean up gpu memory"""
        torch.cuda.empty_cache()
    print('Training Complete!')
    writer.close()
    env.close()
