from typing import Optional

import gym
from meta_critics.running_spec import RunningSpec
from meta_critics.policies.policy import Policy


class SamplerBasedCollector:
    def __init__(self, agent_policy: Policy, spec: RunningSpec, world_size: int, env: Optional[gym.Env] = None):
        """

        :param agent_policy:
        :param spec:
        :param world_size:
        :param env:
        """
        self.spec = spec
        self.world_size = world_size
        self.env_name = self.spec.get('env_name')

        self.env_kwargs = {}
        if hasattr(self.spec, 'env_args'):
            self.env_kwargs = self.spec.env_args

        # batch
        self.batch_size = self.spec.get('num_trajectory', 'meta_task')
        self.agent_policy = agent_policy
        self.seed = None
        self.seed = self.spec.get('seed')

        if env is None:
            env = gym.make(self.env_name, **self.env_kwargs)

        self.env = env
        if self.seed is not None:
            self.env.reset(seed=self.seed)
            self.env.action_space.seed(self.seed)
            self.env.observation_space.seed(self.seed)

        self.env.close()
        self.closed = False

    def sample_async(self, *args, **kwargs):
        """
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    def sample(self, *args, **kwargs):
        """
        :param args:
        :param kwargs:
        :return:
        """
        return self.sample_async(*args, **kwargs)
