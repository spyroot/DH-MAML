#
# This version work with the latest version of gym.
# container separate condition for truncated and terminated.
#
# Mus
from __future__ import annotations

from abc import ABC
from copy import deepcopy
from typing import Iterator, Callable, Optional

import gym
import numpy as np
import torch
from gym import Env
from gym.spaces import Space
from gym.vector import SyncVectorEnv
from gym.vector.utils import create_empty_array

from meta_critics.envs.env_types import EnvType


class VectorizedMetaTask(SyncVectorEnv, ABC):
    def __init__(self,
                 env_fn: Iterator[Callable[[], Env]],
                 observation_space: Space = None,
                 action_space: Space = None,
                 copy: Optional[bool] = True,
                 debug: Optional[bool] = False,
                 **kwargs):
        """
            This vectorized environment that serially runs
            multiple task on a given environments.

        :param env_fn:
        :param observation_space:
        :param action_space:
        :param copy:  if True then reset and step method return a copy of the observation for a task.
        :param kwargs:
        """
        super(VectorizedMetaTask, self).__init__(env_fn,
                                                 observation_space=observation_space,
                                                 action_space=action_space,
                                                 copy=copy,
                                                 **kwargs)

        self.num_task_envs = 0
        self.debug = debug
        self.out = None

        for env in self.envs:
            self.num_task_envs += 1
            if not hasattr(env.unwrapped, 'reset_task'):
                raise ValueError('The environment provided is not a '
                                 'meta-learning environment. It does not have '
                                 'the method `reset_task` implemented.')
        if self.debug:
            print(f"Creating vectorized meta task environment wrapper. "
                  f"total number of envs {self.num_task_envs}")

    def obs_shape(self):
        """"
        """
        if isinstance(self.observation_space, gym.spaces.Discrete):
            obs_shape = (1,)
        elif isinstance(self.observation_space, gym.spaces.Box):
            obs_shape = self.observation_space.shape
        else:
            raise ValueError("Unsupported observation space")
        return obs_shape

    @property
    def action_shape(self):
        if isinstance(self.action_space, gym.spaces.Box):
            action_shape = self.action_space.shape
        elif isinstance(self.action_space, gym.spaces.Discrete):
            action_shape = (1,)
        else:
            raise ValueError("Unsupported action space.")
        return action_shape

    def sample(self) -> np.ndarray | torch.Tensor:
        """ Sampling from action space.
        :returns: Random action from action space return either ndarray or tensor
        """
        if self.out == EnvType.Tensor:
            return torch.from_numpy(self.action_space.sample())
        return self.action_space.sample()


class BaseVecMetaTaskEnv(VectorizedMetaTask, ABC):
    def __init__(self,
                 env_fn: Iterator[Callable[[], Env]],
                 observation_space: Space = None,
                 action_space: Space = None,
                 debug: Optional[bool] = True,
                 copy: Optional[bool] = True,
                 **kwargs):

        super(BaseVecMetaTaskEnv, self).__init__(env_fn,
                                                 observation_space=observation_space,
                                                 action_space=action_space,
                                                 copy=copy,
                                                 debug=debug,
                                                 **kwargs)

    def get_envs(self):
        return self.envs

    def is_terminated(self):
        return self._terminateds.all()

    def is_truncated(self) -> bool:
        return self._truncateds.all()

    def is_done(self) -> bool:
        return self.num_task_envs == np.count_nonzero(self._terminateds) + np.count_nonzero(self._truncateds)

    def get_observations(self):
        """
        Return current observation
        :return:
        """
        return self.observations

    def get_terminated(self):
        """
        Returns all terminated trajectory as vector.
        :return:
        """
        return self._terminateds

    def get_truncated(self):
        """
        Returns all truncated as vector.
        :return:
        """
        return self._truncateds

    def reset_task(self, task):
        """
        Reset task.
        :param task:
        :return:
        """
        for env in self.envs:
            env.unwrapped.reset_task(task)

    def dump_observation_space(self):
        for env in self.envs:
            env.dump_observation_space()

    def step_wait(self):
        """

        """
        observations_list, infos = [], {}
        batch_ids, j = [], 0

        _action = list(self._actions)
        num_actions = len(_action)
        rewards = np.zeros((num_actions,), dtype=self._rewards.dtype)

        terminated_offset = 0
        for i, env in enumerate(self.envs):
            if self._terminateds[i] or self._truncateds[i]:
                continue

            action = _action[j]
            obs, rewards[j], self._terminateds[i], self._truncateds[i], info = env.step(action)
            batch_ids.append(i - terminated_offset)
            j += 1
            if self._terminateds[i] or self._truncateds[i]:
                continue
            observations_list.append(obs)
            infos = self._add_info(infos, info, i)

        # assert num_actions == (j - terminated_offset)
        if observations_list:
            observations = create_empty_array(self.single_observation_space,
                                              n=len(observations_list),
                                              fn=np.zeros)
            assert observations.shape[0] == len(observations_list)
            np.stack(observations_list, axis=0, out=observations)
        else:
            if self.debug:
                print("Observation containers zero elements.")
                print(f"Total number of envs {self.num_envs} total number of terminate and truncated: "
                      f"{np.count_nonzero(self._terminateds) + np.count_nonzero(self._truncateds)}")
            observations = None

        return (deepcopy(self.observations) if self.copy else self.observations,
                rewards,
                np.copy(self._terminateds),
                np.copy(self._truncateds),
                {'batch_ids': batch_ids,
                 'infos': infos})
