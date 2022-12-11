import asyncio
from abc import ABC

from meta_critics.models.gradient_learner import GradientMetaLearner
from meta_critics.policies.policy import Policy
from meta_critics.running_spec import RunningSpec


class AsyncGradientBasedMetaLearner(GradientMetaLearner, ABC):
    def __init__(self, policy: Policy, spec: RunningSpec):
        """

        :param policy:
        :param spec:
        """
        super(AsyncGradientBasedMetaLearner, self).__init__(policy, spec)
        assert self.policy is not None
        assert self.spec is not None
        self._event_loop = asyncio.get_event_loop()

    def _async_gather(self, coroutines):
        """
        :param coroutines:
        :return:
        """
        print(type(coroutines))
        coroutine = asyncio.gather(*coroutines)
        co = zip(*self._event_loop.run_until_complete(coroutine))
        return co
