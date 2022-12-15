from typing import Dict, Optional

import gym
from meta_critics.running_spec import RunningSpec


def create_env_from_spec(running_spec: RunningSpec,
                         render_mode: Optional[str] = None,
                         do_close: Optional[bool] = True,
                         autoreset: bool = False,
                         apply_api_compatibility: Optional[bool] = None,
                         disable_env_checker: Optional[bool] = None,
                         max_episode_steps: Optional[int] = 0) -> gym.Env:
    """Creates a gum environment. This method mainly used to infer observation and action space.
    :param disable_env_checker:
    :param apply_api_compatibility:
    :param max_episode_steps:
    :param running_spec: dh-maml spec.
    :param render_mode:  render mode "human" or none
    :param do_close: if False will not close env, for human rendering.
    :return:
    """
    env_args = None
    if hasattr(running_spec, 'env_args'):
        env_args = running_spec.env_args
        if max_episode_steps > 0:
            env_args["max_episode_steps"] = max_episode_steps

    if env_args is None and max_episode_steps > 0:
        if max_episode_steps > 0:
            env_args["max_episode_steps"] = max_episode_steps

    if hasattr(running_spec, 'env_name'):
        if env_args is None:
            print("Case one")
            env = gym.make(running_spec.env_name, render_mode=render_mode)
        else:
            print("Case two")
            env = gym.make(running_spec.env_name,
                           autoreset=False, apply_api_compatibility=None,
                           disable_env_checker=None, render_mode=render_mode, **env_args)

        if do_close:
            env.close()

    return env


def create_env_from_name(env_name: str, env_args: Dict, render_mode: Optional[str] = None) -> gym.Env:
    """Create environment.
    :return: True if environment created.
    """
    if env_args is None:
        env = gym.make(env_name)
    else:
        env = gym.make(env_name, **env_args)
    env.close()
    return env
