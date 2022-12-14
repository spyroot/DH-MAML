import argparse
import os

import gym

from meta_critics.envs.env_creator import env_creator
from meta_critics.envs.env_vectorized_meta_task import BaseVecMetaTaskEnv
from meta_critics.running_spec import RunningSpec, RunningSpecError
from meta_critics.app_globals import get_running_mode, SpecTypes, AppSelector
from meta_critics.envs.lander.lander import *
from util import create_env_from_name, create_env_from_spec
import torch


def env_only():
    env = gym.make('turbulencelander-v0')
    observation, info = env.reset(seed=42)

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(reward)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()


def vectorized():
    batch_size = 20
    env = gym.make('turbulencelander-v0')
    seed = 1
    envs = BaseVecMetaTaskEnv([env_creator("turbulencelander-v0",
                                           env_kwargs={},
                                           seed=1,
                                           debug=False)
                               for _ in range(batch_size)],
                              observation_space=env.observation_space,
                              action_space=env.action_space,
                              debug=False)
    tasks = env.unwrapped.sample_tasks(4)
    for i, task in enumerate(tasks):
        envs.reset_task(task)

    for i, env in enumerate(envs.get_envs()):
        print(env.task())

    # envs.
    # obs, _ = envs.reset()


def continuous_check_from_name():
    batch_size = 20
    env = create_env_from_name('turbulencelander-v0', {"continuous": True})
    seed = 1
    envs = BaseVecMetaTaskEnv([env_creator("turbulencelander-v0",
                                           env_kwargs={"continuous": True},
                                           seed=1,
                                           debug=False)
                               for _ in range(batch_size)],
                              observation_space=env.observation_space,
                              action_space=env.action_space,
                              debug=False)
    tasks = env.unwrapped.sample_tasks(4)
    for i, task in enumerate(tasks):
        envs.reset_task(task)

    for i, env in enumerate(envs.get_envs()):
        print(env.is_continuous())

    # envs.
    # obs, _ = envs.reset()


def continuous_check_from_spec():
    parser = argparse.ArgumentParser(description="Unit test for threaded observer")
    parser.add_argument('--tune', action='store_true', required=False, help='run ray hyperparameter optimization.')
    parser.add_argument('--test', action='store_true', required=False, help="train model for task")
    parser.add_argument('--plot', action='store_true', required=False, help="test model on task")
    parser.add_argument('--train', action='store_true', required=False, help="plots test result")
    parser.add_argument('--use-cpu', action='store_true', help='if we want enforce cpu only.')
    parser.add_argument('--config', type=str, required=False,
                        default="configs/lander_continuous.yaml",
                        help="a path to the configuration json or yaml file.")
    parser.add_argument('--model_file', type=str, required=False, default="default.th",
                        help="a path to the a model file.")

    # Evaluation
    evaluation = parser.add_argument_group('Evaluation')
    evaluation.add_argument('--num_batches', type=int, default=10, help="number of batches (default: 10)")
    evaluation.add_argument('--num_meta_task', type=int, default=40, help="number of tasks per batch (default: 40)")

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    parser.add_argument('--config_type', type=SpecTypes, default=SpecTypes.JSON, help='config file type.')
    misc.add_argument('--model_dir', type=str, required=False, help='a directory where we will model data.')
    misc.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--debug_agent', action='store_true', required=False, help='Enables debug for agent.')
    parser.add_argument('--debug_env', action='store_true', required=False, help='Enables debug environment.')
    parser.add_argument('--debug_task_sampler', action='store_true', required=False, help='Enables debug environment.')
    # parser = argparse.ArgumentParser(description='PyTorch RPC Batch RL example')
    parser.add_argument('--gamma', type=float, default=1.0, metavar='G', help='discount factor (default: 1.0)')
    #   parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
    parser.add_argument('--num-episode', type=int, default=10, metavar='E', help='number of episodes (default: 10)')
    misc.add_argument('--workers', type=int, default=2, help='Number of workers minimum 2.')
    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()) else 'cpu')

    current_dir = os.getcwd()
    running_spec = RunningSpec(args, AppSelector.TranModel, current_dir)

    batch_size = 20
    env = create_env_from_spec(running_spec)
    seed = 1
    envs = BaseVecMetaTaskEnv([env_creator("turbulencelander-v0",
                                           env_kwargs={"continuous": True},
                                           seed=1,
                                           debug=False)
                               for _ in range(batch_size)],
                              observation_space=env.observation_space,
                              action_space=env.action_space,
                              debug=False)

    tasks = env.unwrapped.sample_tasks(4)
    for i, task in enumerate(tasks):
        envs.reset_task(task)

    for i, env in enumerate(envs.get_envs()):
        print(env.is_continuous())

    # envs.
    # obs, _ = envs.reset()


def env_action():
    env = create_env_from_name('turbulencelander-v0', {"continuous": True})

    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(reward)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()


if __name__ == '__main__':
    try:
        env_action()
    except RunningSpecError as spec_err:
        print(spec_err)
