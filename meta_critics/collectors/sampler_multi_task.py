from abc import ABC
from copy import deepcopy

from meta_critics.running_spec import RunningSpec
from meta_critics.collectors.base.samplerbasedcollector import SamplerBasedCollector
from meta_critics.agents.trajectory.agent_simple import AgentSingleWorker
from meta_critics.policies.policy import Policy


class MultiTaskSampler(SamplerBasedCollector, ABC):

    def __init__(self,
                 spec: RunningSpec,
                 policy: Policy,
                 baseline,
                 env=None):
        """

        :param spec:
        :param policy:
        :param baseline:
        :param env:
        """
        super(MultiTaskSampler, self).__init__(spec, policy, env=env)
        assert self.agent_policy is not None
        print("Creating MultiTaskSampler")

        self.workers = AgentSingleWorker(spec,
                                         self.env.observation_space,
                                         self.env.action_space,
                                         self.agent_policy,
                                         deepcopy(baseline))
        self.task_list = None

    def sample_tasks(self, num_tasks):
        """

        :param num_tasks:
        :return:
        """
        if self.task_list is not None:
            return self.task_list

        tasks = self.env.unwrapped.sample_tasks(num_tasks)
        # tasks = [(index, task) for task in enumerate(tasks)]
        self.task_list = tasks
        return tasks

    def sample(self, tasks):
        """
        :param tasks:
        :return:
        """
        return self.workers.sample(tasks)
