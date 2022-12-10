from abc import ABC, abstractmethod
from typing import Tuple
import torch


class Bandit(ABC):
    """
    """
    @abstractmethod
    def step(self, action: int) -> Tuple[torch.Tensor, int]:
        """
        Do a step and generate reward.
        :param action:
        :return:   Tuple of the next context and the reward generated for given action
        """

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """Reset bandit.
        Returns:
            torch.Tensor: Current context selected by bandit.
        """


class BanditAgent(ABC):
    """Abstract Base class for bandit agents"""
    @abstractmethod
    def receive_reward(self, reward: int) -> None:
        """

        :param reward:
        :return:
        """
        pass

    @abstractmethod
    def select_action(self, context: torch.Tensor) -> int:
        """
        Select an action based either in multi armed or contextual arm bandits
        :param context:
        :return:
        """
        pass
