import argparse
import gym
import sys
import itertools
from RL.models.policy import ReparameterGaussianPolicy
from RL.models.critic import ActionCritic
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from RL.utils import *


"""
 parameters
"""
parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default="Hopper-v2", help='name of the environment to run')
parser.add_argument('--max_steps', type=int, default=1000001, help="the total sample steps in env")
parser.add_argument('--batch_size', type=int, default=512, help="batch learning size")
parser.add_argument('--gradient_step', type=int, default=1, help="gradient descent times per learning")
parser.add_argument('--gamma', type=float, default=0.99, help="the discount factor")
parser.add_argument('--learning_rate', type=float, default=3e-4, help="learning rate for all")
parser.add_argument('--tau', type=float, default=0.005, help='soft update coefficient')
parser.add_argument('--hidden_size', type=int, default=256, help='hidden layer size for all NN')
parser.add_argument('--seed', type=int, default=0, help="random seed")
parser.add_argument('--eval', type=bool, default=True, help="whether evaluate the policy")
parser.add_argument('--target_update_interval', type=int, default=1,
                    help="Q network update interval for target Q network")
parser.add_argument('--save_model_interval', type=int, default=1000000, help="step interval for saving model")
args = parser.parse_args()
# alpha = 1.0

"""
 environment
"""
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

"""
 model
"""
policy = ReparameterGaussianPolicy(state_dim, action_dim, args.hidden_size, env.action_space).to(device)
q_netowrk_1 = ActionCritic(state_dim, action_dim).to(device)
q_netowrk_2 = ActionCritic(state_dim, action_dim).to(device)
target_q_network_1 = ActionCritic(state_dim, action_dim).to(device)
target_q_network_2 = ActionCritic(state_dim, action_dim).to(device)
hard_update(target_q_network_1, q_netowrk_1)
hard_update(target_q_network_2, q_netowrk_2)

"""
 optimizer
"""
optimizer_q1 = optim.Adam(q_netowrk_1.parameters(), lr=args.learning_rate)
optimizer_q2 = optim.Adam(q_netowrk_2.parameters(), lr=args.learning_rate)
optimizer_p = optim.Adam(policy.parameters(), lr=args.learning_rate)
loss_func = nn.MSELoss()

"""
 others
"""
memory = Memory(1e6)
env.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
writer = SummaryWriter('sac-Hopper-v2')


def evaluate_policy(trajectory_sum):
    state = env.reset()
    state = torch.tensor([state]).float().to(device)
    log = {}
    total_steps = 0
    trajectory_reward = 0.0
    traj = 1

    while traj <= trajectory_sum:
        with torch.no_grad():
            # _, _, action = policy.rsample(state)
            action = policy.select_action(state)
        next_state, reward, done, _ = env.step(action[0].cpu().numpy())
        log['total_reward'] = log.get('total_reward', 0) + reward
        trajectory_reward += reward
        total_steps += 1
        if done:
            traj += 1
            log['min_traj_reward'] = min(log.get('min_traj_reward', sys.maxsize), trajectory_reward)
            log['max_traj_reward'] = max(log.get('max_traj_reward', -sys.maxsize), trajectory_reward)
            trajectory_steps, trajectory_reward = 0, 0.0
            state = env.reset()
            state = torch.tensor([state]).float().to(device)
        else:
            state = next_state
            state = torch.tensor([state]).float().to(device)
    log['avg_traj_reward'] = log['total_reward'] / traj
    log['avg_traj_step'] = total_steps / traj
    log['reward_per_step'] = log['total_reward'] / total_steps
    return log


def sac_step(update):
    for u in range(args.gradient_step):
        batch = memory.sample(args.batch_size)
        state = torch.cat(batch.state, 0)
        action = torch.cat(batch.action, 0)
        reward = torch.cat(batch.reward, 0).unsqueeze(1)
        next_state = torch.cat(batch.next_state, 0)
        mask = torch.cat(batch.mask, 0).unsqueeze(1)

        # compute targets for the Q functions
        with torch.no_grad():
            # next_action, next_log_pi, _ = policy.rsample(next_state)
            next_action = policy.select_action(next_state)
            next_log_pi = policy.log_prob(next_state, next_action)
            nsa = torch.cat([next_state, next_action], 1)
            target = reward + args.gamma * mask * \
                     (torch.min(target_q_network_1(nsa), target_q_network_2(nsa))
                      - next_log_pi)

        # update Q-functions by one step of gradient descent
        for net, optimizer in [(q_netowrk_1, optimizer_q1), (q_netowrk_2, optimizer_q2)]:
            optimizer.zero_grad()
            predict = net(torch.cat([state, action], 1))
            mse_loss = loss_func(predict, target)
            mse_loss.backward()
            optimizer.step()

        # update policy by one step of gradient descent

        # 这里是resample版的
        sample_action, log_pi, _ = policy.rsample(state)
        sa = torch.cat([state, sample_action], 1)
        optimizer_p.zero_grad()
        kl_loss = (log_pi - torch.min(q_netowrk_1(sa), q_netowrk_2(sa))).mean()

        # 这里是gradient estimator版的
        # sample_action = policy.select_action(state)
        # log_pi = policy.log_prob(state, sample_action)
        # with torch.no_grad():
        #     log_pi_con = policy.log_prob(state, sample_action)
        # sa = torch.cat([state, sample_action], 1)
        # optimizer_p.zero_grad()
        # kl_loss = (log_pi * (log_pi_con - torch.min(q_netowrk_1(sa), q_netowrk_2(sa)) + 1)).mean()
        kl_loss.backward()
        optimizer_p.step()

        # update target network
        if update % args.target_update_interval == 0:
            soft_update(target_q_network_1, q_netowrk_1, args.tau)
            soft_update(target_q_network_2, q_netowrk_2, args.tau)


def main_loop():
    update = 0
    total_steps = 0
    for i_episode in itertools.count(1):
        state = env.reset()
        state = torch.tensor([state]).float().to(device)
        done = False

        while not done:
            with torch.no_grad():
                # action, _, _ = policy.rsample(state)  # Sample action from policy
                action = policy.select_action(state)
            next_state, reward, done, _ = env.step(action[0].cpu().numpy())
            next_state = torch.tensor([next_state]).float().to(device)
            reward = torch.tensor([reward]).to(device)
            mask = torch.tensor([0.0] if done else [1.0]).to(device)
            memory.complex_push(state, action, reward, next_state, mask, 0)
            total_steps += 1
            if total_steps >= args.batch_size:
                update += 1
                sac_step(update)
            state = next_state

        # evaluate the policy
        if i_episode % 10 == 0 and eval is True:
            print(f"Begin episode{i_episode} evaluating, total_steps:{total_steps}-----------------------------")
            log = evaluate_policy(20)
            print(f"total_reward:{log['total_reward']: .2f}, "
                  f"avg_traj_reward:{log['avg_traj_reward'] : .2f}, "
                  f"avg_traj_step:{log['avg_traj_step']: .2f}, "
                  f"max_traj_reward:{log.get('max_traj_reward', log['total_reward']): .2f}, "
                  f"per_step_reward:{log['reward_per_step']: .2f}")

        # save the model
        if total_steps % args.save_model_interval == 0:
            torch.save(policy.state_dict(), 'nets/policy_sac.pkl')
            torch.save(q_netowrk_1.state_dict(), 'nets/q1.pkl')
            torch.save(q_netowrk_2.state_dict(), 'nets/q2.pkl')

        if total_steps >= args.max_steps:
            break

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main_loop()

