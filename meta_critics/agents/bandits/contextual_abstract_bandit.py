from abc import ABC
from typing import List, Union
from meta_critics.agents.bandits.bandit_agent import Bandit
import torch


class MultiArmedBandit(Bandit, ABC):
    """
    Base Class for a Contextual Multi-armed Bandit
    :param bandits: Number of bandits
    :param arms: Number of arms in each bandit
    :param context_type: Give context as either tensor or int
    :type bandits: int
    :type arms: int
    :type context_type: str
    """

    def __init__(self, bandits: int = 1, arms: int = 1, context_type: str = "tensor"):
        if not (context_type == "int" or context_type == "tensor"):
            raise ValueError(f"context_type should be either tensor or int, found {context_type}")

        self.context_type = context_type
        self._reset_metrics()

    @property
    def reward_hist(self) -> List[float]:
        """
        Get the history of rewards received at each step
        :returns: List of rewards
        :rtype: list
        """
        return self._reward_hist

    @property
    def regret_hist(self) -> List[float]:
        """
        Get the history of regrets incurred at each step
        :returns: List of regret
        :rtype: list
        """
        return self._regret_hist

    @property
    def cum_regret_hist(self) -> Union[List[int], List[float]]:
        return self._cum_regret_hist

    @property
    def cum_reward_hist(self) -> Union[List[int], List[float]]:
        return self._cum_reward_hist

    @property
    def cum_regret(self) -> Union[int, float]:
        return self._cum_regret

    @property
    def cum_reward(self) -> Union[int, float]:
        return self._cum_reward

    @property
    def arms(self) -> int:
        """
        Get the number of arms in each bandit
        :returns: Number of arms in each bandit
        :rtype: int
        """
        return self._narms

    @property
    def bandits(self) -> int:
        """
        Get the number of bandits
        :returns: Number of bandits
        :rtype: int
        """
        return self._nbandits

    def _reset_metrics(self) -> None:
        """
        Resets the various metrics to empty
        """
        self._regret_hist = []
        self._reward_hist = []
        self._cum_regret_hist = []
        self._cum_reward_hist = []
        self._cum_regret = 0
        self._cum_reward = 0

    def _reset_bandit(self) -> None:
        """
        Resets the current bandit and context
        """
        self.curr_bandit = torch.randint(self.bandits, (1,))
        self.curr_context = F.one_hot(self.curr_bandit, num_classes=self.context_dim).to(torch.float)

    def reset(self) -> torch.Tensor:
        """
        Resets metrics to empty the current bandit randomly
        :returns: The current bandit as observation
        :rtype: int
        """
        self._reset_metrics()
        self._reset_bandit()
        if self.context_type == "tensor":
            return self.curr_context.view(-1)
        elif self.context_type == "int":
            return self.curr_bandit.item()
    #
    # def update_metric(self, reward: int,
    #                   max_reward: int) -> Tuple[Union[int, torch.Tensor], Union[int, float]]:
    #     """
    #     Takes an action in the bandit and returns the sampled reward
    #     This method needs to be implemented in the specific bandit.
    #
    #     :param max_reward:
    #     :param reward:
    #     :returns: Reward sampled for the action taken
    #     :rtype: int, float ...
    #     """
    #     regret = max_reward - reward
    #     self._cum_regret += regret
    #     self.cum_regret_hist.append(self._cum_regret)
    #     self.regret_hist.append(regret)
    #     self._cum_reward += reward
    #     self.cum_reward_hist.append(self._cum_reward)
    #     self.reward_hist.append(reward)
    #     self._reset_bandit()
    #
    #     if self.context_type == "tensor":
    #         return self.curr_context.view(-1), reward
    #     elif self.context_type == "int":
    #         return self.curr_bandit, reward
