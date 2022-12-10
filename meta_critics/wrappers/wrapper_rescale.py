from abc import ABC
from typing import Union

import gym
import numpy as np
from gym.core import ActionWrapper
from gym.spaces import Box
from gym.vector import VectorEnv


class ClipAction(ActionWrapper, ABC):
    """It actions  Wrapper to clip actions, I just clip in low ad high
    :param env: The environment whose actions need to be clipped
    :type env: object
    """
    def __init__(self, env: Union[gym.Env, VectorEnv]):
        """

        :param env:
        """
        super(ClipAction, self).__init__(env)
        assert isinstance(self.env.action_space, Box)

    def action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.env.action_space.low,
                       self.env.action_space.high)


class RescaleAction(ActionWrapper, ABC):
    """Rescale actions
    :param env: The environment whose actions need to be rescaled
    :param low: Lower limit of action
    :param high: Upper limit of action
    :type env: object
    :type low: int
    :type high: int
    """

    def __init__(self, env: Union[gym.Env, VectorEnv], low: int, high: int):
        super(RescaleAction, self).__init__(env)
        assert isinstance(self.env.action_space, Box)
        assert high > low

        self.low = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + low
        self.high = (
                np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + high
        )
        self.action_space = Box(
                low=low,
                high=high,
                shape=env.action_space.shape,
                dtype=env.action_space.dtype,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        :param action:
        :return:
        """
        assert np.all(action >= self.low)
        assert np.all(action <= self.high)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low) * ((action - self.low) / (self.high - self.low))
        return np.clip(action, low, high)
