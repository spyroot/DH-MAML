from typing import Dict

import gym
from meta_critics.running_spec import RunningSpec


def create_env_from_spec(running_spec: RunningSpec) -> gym.Env:
    """Create environment.
    :return: True if environment created.
    """
    env_args = None
    if hasattr(running_spec, 'env_args'):
        env_args = running_spec.env_args

    if hasattr(running_spec, 'env_name'):
        if env_args is None:
            env = gym.make(running_spec.env_name)
        else:
            env = gym.make(running_spec.env_name, **env_args)
        env.close()

    return env


def create_env_from_name(env_name: str, env_args: Dict) -> gym.Env:
    """Create environment.
    :return: True if environment created.
    """
    if env_args is None:
        env = gym.make(env_name)
    else:
        env = gym.make(env_name, **env_args)
    env.close()
    return env
