#
# This main env created.  It called in many places,
#
# It creates env and intercept call to __call__  so we can reset
# seed for different vectorized env.
#
# Mus
from typing import Optional
import gym
import torch


class env_creator(object):
    def __init__(self,
                 env_name: Optional[str],
                 env_kwargs=None,
                 seed: Optional[int] = None,
                 debug: Optional[bool] = False,
                 max_episode_steps: Optional[int] = None,
                 device: Optional[torch.device] = 'cpu',
                 check_env: Optional[bool] = "False"):
        """
        Create environment and set fixed seed to each.
        :param env_name:  gym_id
        :param env_kwargs: args require for env
        :param seed:
        """
        if env_kwargs is None:
            env_kwargs = {}

        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.debug = debug
        self.seed = seed
        self.device = device
        self.check_env = check_env
        # print(f"env_creator: creating environment device {self.device}, "
        #       f"seed: {self.seed} env name: {self.env_name} env args {self.env_kwargs}")

    def __call__(self):
        """ Intercept call to env and resets seed.
        :return:
        """
        env = gym.make(self.env_name, disable_env_checker=self.check_env, **self.env_kwargs)
        if self.seed is not None:
            # print(f"Resting env {self.env_name} with a seed {self.seed}")
            env.reset(seed=self.seed)
            env.action_space.seed(self.seed)
            env.observation_space.seed(self.seed)
        return env
