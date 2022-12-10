from abc import ABC, abstractmethod
from typing import Optional

import gym
from torch.distributed import rpc

from meta_critics.collectors.base.samplerbasedcollector import SamplerBasedCollector
from meta_critics.policies.policy import Policy
from meta_critics.running_spec import RunningSpec


class GenericRpcAgent(SamplerBasedCollector, ABC):
    def __init__(self, agent_policy: Policy, spec: RunningSpec, world_size: int, env: Optional[gym.Env] = None):
        super(GenericRpcAgent, self).__init__(agent_policy, spec, world_size, env=env)

    @staticmethod
    def remote_method(method, rref, *_args, **kwargs):
        args = [method, rref] + list(_args)
        return rpc.rpc_sync(rref.owner(), GenericRpcAgent.call_method, args=args, kwargs=kwargs)

    @staticmethod
    def call_method(method, rref, *args, **kwargs):
        return method(rref.local_value(), *args, **kwargs)

    @abstractmethod
    def rpc_sync_grad(self, worker_id, parameters):
        pass

    @abstractmethod
    def sync_policy(self, worker_id, parameters):
        pass

    @abstractmethod
    def broadcast_grads(self):
        pass

    @abstractmethod
    def shutdown_observers(self):
        pass
