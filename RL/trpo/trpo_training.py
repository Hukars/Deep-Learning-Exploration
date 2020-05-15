import gym
import argparse

from RL.utils import *
from RL.models.value import Value
from RL.models.policy_disc_action import PolicyDisc
from RL.models.policy_cont_action import PolicyCont
from RL.models.agent import Agent

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=500, help="the training epoch")
parser.add_argument('--render', type=bool, default=True, help="whether render the environment")
parser.add_argument('--num_thread', type=int, default=6, help="number of threads for agent (default: 6)")
parser.add_argument('--gamma', type=float, default=0.99, help="the discounted factor")
parser.add_argument('--lambda_', type=float, default=0.95, help="the GAE factor")
parser.add_argument('--batch_size', type=int, default=2048, help="a approximated batch size for batch training")
parser.add_argument('--max_kl', type=float, default=0.2, help='the constrained KL divergence value')
parser.add_argument('--l2_reg', type=float, default=1e-3, help='the factor for L2 regularization')
parser.add_argument('--damping', type=float, default=1e-2, metavar='G', help='damping (default: 1e-2)')
args = parser.parse_args()


def main():

    # environmenr
    env = gym.make("InvertedPendulum-v2")
    state_dim = env.observation_space.shape[0]
    is_disc_action = len(env.action_space.shape) == 0
    running_state = ZFilter((state_dim,), clip=5.0)

    # model
    value_net = Value(state_dim)
    if is_disc_action:
        policy_net = PolicyDisc(state_dim, env.action_space.n).to(device)
    else:
        policy_net = PolicyCont(state_dim, env.action_space.shape[0]).to(device)
    agent = Agent(env, policy_net, running_state, args.render, args.num_thread)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    # train the model
    for epoch in range(args.epoch):
        batch, log = agent.process_sample(args.batch_size)
        states = torch.from_numpy(np.stack(batch.state)).float().to(device)
        actions = torch.from_numpy(np.stack(batch.action)).float().to(device)
        masks = torch.from_numpy(np.stack(batch.mask)).float().to(device)
        rewards = torch.from_numpy(np.stack(batch.reward)).float().to(device)

        # actor-critic baseline
        with torch.no_grad():
            values = value_net(states)

        # use GAE to estimate A(s,a)
        advantages, returns = estimate_advantages(rewards, values, masks, args.gamma, args.lambda_, device)

        # make a optimization step
        trpo_step(policy_net, value_net, states, actions, returns, advantages, args.max_kl, args.damping, args.l2_reg)


if __name__ == "__main__":
    main()