"""
Generic trainer for bandit problems
Author Mustafa
mbayramo@stanford.edu
"""
from abc import abstractmethod, ABC
from typing import Any

from meta_critics.envs.bandits.bandit_base_env import BanditEnv


class BanditTrainer(ABC):
    def __init__(self, agent: Any,
                 bandit_env: BanditEnv,
                 logdir: str = "./logs"):
        """
        :param agent:  Agents , classical agents i.e Epsilon Greedy, Thompson Sampling UCB etc.
        :param bandit_env:  Bandit environ that we use for Meta training/testing.
        :param logdir:
        """
        self.agent = agent
        self.bandit_env = bandit_env
        self.logdir = logdir

    @abstractmethod
    def train(self) -> None:
        pass
