"""
Lazy frame wrapper. Efficient data structure to save each frame only once.
Can use LZ4 compression to optimizer memory usage.
"""
from collections import deque
from typing import Tuple, Optional
import numpy as np
import gym
from gym.core import Wrapper, ObsType
from gym.spaces import Box
from meta_critics.wrappers.generics.lazy_frame import LazyFrames


class FrameStack(Wrapper):
    def __init__(self, env: gym.Env, framestack: Optional[int] = 4, compress: Optional[bool] = True):
        """Wrapper to stack the last few(4 based on regular DQN implementation by default)
        observations of agent efficiently,
        :param env: gym or gym like environment to be wrapped
        :param framestack: number of frames to stack
        :param compress: compress each frame
        """
        super(FrameStack, self).__init__(env)

        self.env = env
        self._frames = deque([], maxlen=framestack)
        self.framestack = framestack

        assert hasattr(self.env.observation_space.low)
        assert hasattr(self.env.observation_space.high)

        low = np.repeat(np.expand_dims(self.env.observation_space.low, axis=0), framestack, axis=0)
        high = np.repeat(np.expand_dims(self.env.observation_space.high, axis=0), framestack, axis=0)

        self.observation_space = Box(low=low, high=high, dtype=self.env.observation_space.dtype)

    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, bool, dict]:
        """Steps through environment
        :param action: Action taken by agent
        :type action: NumPy Array
        :returns: Next state, reward, done, info
        :rtype: NumPy Array, float, boolean, dict
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(observation)
        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, *,
              seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        """Resets environment
        :returns: Initial state of environment
        :rtype: NumPy Array
        """
        observation = self.env.reset()
        for _ in range(self.framestack):
            self._frames.append(observation)
        return self._get_obs(), self.info

    def _get_obs(self) -> np.ndarray:
        """
        Gets observation given deque of frames
        :returns: Past few frames
        :rtype: NumPy Array
        """
        return np.array(LazyFrames(list(self._frames)))[np.newaxis, ...]
