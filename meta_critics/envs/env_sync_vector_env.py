#
# This version work with the latest version of gym.
# container separate condition for truncated and terminated.
#
# Mus
from __future__ import annotations

from abc import ABC
from typing import Iterator, Callable, Optional, Union, List

import gym
import numpy as np
import torch
from gym import Env, Space
from gym.vector.utils import create_empty_array

from meta_critics.envs.env_types import EnvType
from meta_critics.envs.sync_vector_env import SyncVectorEnv


class BaseSyncVectorEnv2(SyncVectorEnv, ABC):
    def __init__(self,
                 env_fns: Iterator[Callable[[], Env]],
                 observation_space: Space = None,
                 action_space: Space = None,
                 copy: Optional[bool] = True,
                 out: Optional[EnvType] = EnvType.NdArray,
                 **kwargs):
        """
        Base vector environment.
        :param env_fns:  Iterator that emits callback, each callback emit gym or gym like env
        :param observation_space: vectorized observation space
        :param action_space:  vectorized action space
        :param copy: return copy during step process.
        :param out: datatype we operate in. If type torch for example step and reset must return torch tensor
        :param kwargs:
        """
        super(BaseSyncVectorEnv2, self).__init__(env_fns,
                                                 observation_space=observation_space,
                                                 action_space=action_space,
                                                 copy=copy,
                                                 **kwargs)
        # out type
        self.out = out
        self.num_task_envs = 0
        for env in self.envs:
            self.num_task_envs += 1
            if not hasattr(env.unwrapped, 'reset_task'):
                raise ValueError('The environment provided is not a '
                                 'meta-learning environment. It does not have '
                                 'the method `reset_task` implemented.')

    def sample(self) -> np.ndarray | torch.Tensor:
        """ Sampling from action space.
        :returns: Random action from action space return either ndarray or tensor
        """
        if self.out == EnvType.Tensor:
            return torch.from_numpy(self.action_space.sample())
        return self.action_space.sample()

    def obs_shape(self):
        """Return shape of observation
        :return:
        """
        if isinstance(self.observation_space, gym.spaces.Discrete):
            obs_shape = (1,)
        elif isinstance(self.observation_space, gym.spaces.Box):
            obs_shape = self.observation_space.shape
        else:
            raise ValueError("Unsupported observation space")
        return obs_shape

    def action_shape(self):
        """Return shape of action.
        :return:
        """
        if isinstance(self.action_space, gym.spaces.Box):
            action_shape = self.action_space.shape
        elif isinstance(self.action_space, gym.spaces.Discrete):
            action_shape = (1,)
        else:
            raise ValueError("Unsupported action space")
        return action_shape

    def reset(self, *, seed: Optional[Union[int, List[int]]] = None,
              options: Optional[dict] = None):
        """Default reset implementation.
        :param seed:
        :param options:
        :return:
        """
        super().reset(seed=seed)
        if self.render_mode == "human":
            self.render()
        return self.observation_space.sample(), {}

    def is_done(self):
        return self.num_task_envs == np.count_nonzero(self._terminateds) + np.count_nonzero(self._truncateds)

    def is_terminated(self) -> bool:
        """Return true if terminated
        :return:
        """
        return self._terminateds.all()

    def is_truncated(self) -> bool:
        """Return true if truncated
        :return:
        """
        return self._truncateds.all()

    def reset_task(self, task) -> None:
        """
        Reset task
        :param task:
        :return:
        """
        for env in self.envs:
            env.unwrapped.reset_task(task)

    def step_wait(self):
        """
        """
        observations_list, infos = [], []
        batch_ids, j = [], 0

        _action = list(self._actions)
        num_actions = len(_action)
        rewards = np.zeros((num_actions,), dtype=np.float_)

        for i, env in enumerate(self.envs):
            if self._terminateds[i] or self._truncateds[i]:
                continue

            action = _action[j]
            obs, rewards[j], self._terminateds[i], self._truncateds[i], info = env.step(action)
            batch_ids.append(i)

            if not self._terminateds[i] and not self._truncateds[i]:
                observations_list.append(obs)
                infos.append(info)
            j += 1

        assert num_actions == j
        if observations_list:
            observations = create_empty_array(self.single_observation_space,
                                              n=len(observations_list),
                                              fn=np.zeros)
            assert observations.shape[0] == len(observations_list)
            np.stack(observations_list, axis=0, out=observations)
        else:
            print("return None")
            observations = None

        return (observations,
                rewards,
                np.copy(self._terminateds),
                np.copy(self._truncateds),
                {'batch_ids': batch_ids, 'infos': infos})
