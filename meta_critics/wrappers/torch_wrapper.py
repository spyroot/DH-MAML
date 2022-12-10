"""This wrapper matches GYM API,  so we can add on stack of wrappers
(i.e last one so each step return tensor not numpy data)

Mus
"""
from typing import Any, Tuple
import gym
import torch


class TorchWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, *args, **kwargs):
        """
        :param env:
        :param args:
        :param kwargs:
        """
        super(TorchWrapper, self).__init__(env, **kwargs)

    def step(self, action: torch.Tensor) -> torch.Tensor:
        """

        :param action:
        :return:
        """
        if self.action_shape == (1,) and isinstance(
            self.env.action_space, gym.spaces.Discrete
        ):
            state, reward, done, truncated, info = self.env.step(action.item())
        else:
            state, reward, done, truncated, info = self.env.step(action.data)
        state = torch.from_numpy(state)
        return state, reward, done, truncated, info

    def reset(self, **kwargs) -> Tuple[torch.Tensor, dict]:
        """

        :param kwargs:
        :return:
        """
        observations, info = super().reset(**kwargs)
        return torch.from_numpy(self.env.reset()), info

    def sample(self) -> torch.Tensor:
        """
        :return:
        """
        return torch.from_numpy(self.env.action_space.sample())

    def __getattr__(self, name: str) -> Any:
        """All other calls would go to base env
        """
        env = super(TorchWrapper, self).__getattribute__("env")
        return getattr(env, name)
