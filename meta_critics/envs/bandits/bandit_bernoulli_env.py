"""Multi-armed bandit problems with Bernoulli observations, as described
   in [1].

   At each time step, the agent pulls one of the `k` possible arms (actions),
   say `i`, and receives a reward sampled from a Bernoulli distribution with
   parameter `p_i`. The multi-armed bandit tasks are generated by sampling
   the parameters `p_i` from the uniform distribution on [0, 1].

   [1] Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever,
       Pieter Abbeel, "RL2: Fast Reinforcement Learning via Slow Reinforcement
       Learning", 2016 (https://arxiv.org/abs/1611.02779)
   """
from abc import ABC
from typing import Tuple, Optional

import numpy as np
from gym.core import ObsType
from gym.vector.utils import spaces
from meta_critics.envs.bandits.bandit_base_env import BanditEnv
from meta_critics.envs.env_types import EnvType


class BernoulliBanditEnv(BanditEnv, ABC):
    def __init__(self, k: int,
                 max_reward: Optional[int] = 1,
                 out: Optional[EnvType] = EnvType.NdArray,
                 task=None):
        """

        :param k:
        :param max_reward:
        :param out:
        :param task:
        """
        super(BernoulliBanditEnv, self).__init__(k=k, max_reward=max_reward, out=out)
        assert self.k > 0
        assert self.max_reward() > 0
        self.action_space = spaces.Discrete(self.k)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        if task is None:
            task = {}

        # print(f"Creating BernoulliBanditEnv task {task}, k={k}")
        self._task = task
        self._means = task.get('mean', np.full((k,), 0.5, dtype=np.float32))

    def sample_tasks(self, num_tasks):
        """ Sample task.
        :param num_tasks:
        :return:
        """
        # self.np_random.random(num_tasks, self.k)
        means = self.np_random.random((num_tasks, self.k))
        tasks = [{'mean': mean} for mean in means]
        return tasks

    def reset_task(self, task):
        """Reset task to bernoulli mean. Value taken from task.
           Task must container key 'mean'
        :param task:
        :return:
        """
        self._task = task
        self._means = task['mean']

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        """Reset environment
        :param seed:
        :param options:
        :return:
        """
        return np.random.uniform(size=1).astype(np.float32), {'task': self._task}

    def step(self, action) -> Tuple[ObsType, float, bool, bool, dict]:
        """Agent takes agent environment return observation, reward, etc
        :param action:
        :return:
        """
        assert self.action_space.contains(action)
        mean = self._means[action]
        reward = self.np_random.binomial(1, mean)
        observation = np.random.uniform(size=1).astype(np.float32)
        return observation, reward, True, False, {'task': self._task}
