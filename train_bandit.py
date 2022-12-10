import argparse
import json
import multiprocessing as mp
import os

import torch
import yaml

from meta_critics.agents.bandits.agent_ucb import UCBAgent
from meta_critics.agents.trainer.bandit_trainer import MultiArmBanditTrainer
from meta_critics.agents.bandits.agent_eps_greedy import EpsGreedyMABAgent


def train_ucb(env):
    # observation, info = env.reset(seed=42)
    agent = UCBAgent(env)
    trainer = MultiArmBanditTrainer(agent=agent, bandit_env=env)
    trainer.train()


def train_epsilon_greedy(env):
    # observation, info = env.reset(seed=42)
    agent = EpsGreedyMABAgent(env)
    trainer = MultiArmBanditTrainer(agent=agent, bandit_env=env, log_freq=0, time_steps=100)
    metrics = trainer.train()

    # for _ in range(1000):
    #     action = env.action_space.sample()
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     print(reward)
    #     if terminated or truncated:
    #         observation, info = env.reset()
    # env.close()


def main(args):
    """

    :param args:
    :return:
    """
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.output_folder is not None:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        policy_filename = os.path.join(args.output_folder, 'policy.th')
        config_filename = os.path.join(args.output_folder, 'config.json')

        with open(config_filename, 'w') as f:
            config.update(vars(args))
            json.dump(config, f, indent=2)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta Reinforcement learning with')
    parser.add_argument('--config', type=str, required=False, help='configuration file.')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output-folder', type=str, help='name of the output folder')
    misc.add_argument('--seed', type=int, default=None, help='random seed')
    misc.add_argument('--use-cuda', action='store_true',
                      help='use cuda (default: false, use cpu). WARNING: Full support for cuda '
                           'is not guaranteed. Using CPU is encouraged.')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
                      help='number of workers for trajectories sampling (default: '
                           '{0})'.format(mp.cpu_count() - 1))
    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available() and args.use_cuda) else 'cpu')
    main(args)
