"""This wrapper that tracks the cumulative rewards and episode
lengths and serialize data to tensorboard.

So we can track what agent is doing.

Mus
"""
import time
from collections import deque
from typing import Optional, Callable, Tuple
import numpy as np
import gym
from gym.core import ObsType
from torch.utils.tensorboard import SummaryWriter


def add_vector_episode_statistics(info: dict, episode_info: dict, num_envs: int, env_num: int):
    """Adds episode statistics.
    Add statistics coming from the vectorized environment.
    Args:
        info (dict): info dict of the environment.
        episode_info (dict): episode statistics data.
        num_envs (int): number of environments.
        env_num (int): env number of the vectorized environments.
    Returns:
        info (dict): the input info dict with the episode statistics.
    """
    info["episode"] = info.get("episode", {})
    info["_episode"] = info.get("_episode", np.zeros(num_envs, dtype=bool))
    info["_episode"][env_num] = True

    for k in episode_info.keys():
        info_array = info["episode"].get(k, np.zeros(num_envs))
        info_array[env_num] = episode_info[k]
        info["episode"][k] = info_array

    return info


class RecordTensorboardEpisodeStatistics(gym.Wrapper):
    """
    """
    def __init__(self, env: gym.Env, writer: SummaryWriter, is_eval_callback: Callable, deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths
        and serialize data to tensorboard log.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.t0 = time.perf_counter()
        self.episode_count = 0
        self.episode_returns: Optional[np.ndarray] = None
        self.episode_lengths: Optional[np.ndarray] = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.writer = writer
        self.global_step = 0
        self.is_eval_callback = is_eval_callback

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        """Resets the environment,  episode returns and lengths."""
        observations, info = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations, info

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(action)
        assert isinstance(infos, dict), \
            f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to " \
            f"usage of other wrappers in the wrong order."
        self.episode_returns += rewards
        self.episode_lengths += 1

        self.global_step += 1 * self.num_envs

        if not self.is_vector_env:
            terminateds = [terminateds]
            truncateds = [truncateds]
        terminateds = list(terminateds)
        truncateds = list(truncateds)

        for i in range(len(terminateds)):
            if terminateds[i] or truncateds[i]:
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                print(type(terminateds))
                # print(episode_return)
                # print(episode_return)

                if self.is_eval_callback is not None and self.is_eval_callback():
                    self.writer.add_scalar("evaluate/episodic_return", episode_return,  global_step=self.global_step)
                    self.writer.add_scalar("evaluate/episodic_length", episode_length, global_step=self.global_step)
                else:
                    self.writer.add_scalar("train/episodic_return", episode_return, self.global_step)
                    self.writer.add_scalar("train/episodic_length", episode_length, self.global_step)

                self.return_queue.append(episode_return)
                self.length_queue.append(episode_length)
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        return (
            observations,
            rewards,
            terminateds if self.is_vector_env else terminateds[0],
            truncateds if self.is_vector_env else truncateds[0],
            infos,
        )
