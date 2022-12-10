"""
Used to compare result similarly RL^2
Base class for Multi armed or Contextual Bandits
Mus
# mbayramo@stanford.edu
"""
from typing import Tuple, Union, List
import numpy as np
from meta_critics.agents.bandits.bandit_agent import BanditAgent
from meta_critics.envs.bandits.bandit_base_env import BanditEnv


class MultiArmAgent(BanditAgent):
    def __init__(self, env_bandit: BanditEnv):
        """
        :param env_bandit:
        """
        super(MultiArmAgent, self).__init__()
        if env_bandit is None:
            raise ValueError("bandit environment is none.")

        # environment gym or gym like
        self._env_bandit = env_bandit
        self._arm_counts = np.zeros(shape=(env_bandit.observation_space.shape[0],
                                           self._env_bandit.get_num_arms()))
        self._regret = 0.0

        # history
        self._regret_hist = []
        self._action_hist = []
        self._reward_hist = []

        # accumulated stats
        self._cum_regret_hist = []
        self._cum_reward_hist = []
        self._cum_regret = 0
        self._cum_reward = 0

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._regret_hist = []
        self._reward_hist = []
        self._cum_regret_hist = []
        self._cum_reward_hist = []
        self._cum_regret = 0
        self._cum_reward = 0

    @property
    def action_history(self) -> List[Tuple[int, int]]:
        """ The history of actions taken for contexts
        :returns: List of context, actions pairs
        """
        return self._action_hist

    @property
    def regret(self) -> float:
        """ Current regret
        :returns: The current regret
        """
        return self._regret

    @property
    def regret_history(self) -> List[float]:
        """Get the history of regrets incurred for each step
        :returns: List of rewards
        """
        return self._regret_hist

    @property
    def reward_history(self) -> List[float]:
        """history of rewards received for each step
        :returns: List of rewards
        """
        return self._reward_hist

    @property
    def counts(self) -> np.ndarray:
        """Number of times each action has been taken
        :returns: Numpy array with count for each action
        """
        return self._arm_counts

    def select_action(self, context: int) -> int:
        """Select an action
        :param context: the context to select action for
        :returns: Selected action
        """
        raise NotImplementedError

    def update_params(self, context: int, action: int, reward: Union[int, float]) -> None:
        """Updates parameters for the policy.
        :param context: context for which action is taken in multi armed bandit single context.
        :param action: action taken for the step
        :param reward: reward obtained.
        """
        raise NotImplementedError

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

    def receive_reward(self, reward: int) -> None:
        """Receives reward for taking and action.
        :param reward: received reward.
        :return:
        """
        regret = self._env_bandit.max_reward() - reward
        self._cum_regret += regret
        self.cum_regret_hist.append(self._cum_regret)
        self.regret_history.append(regret)
        self._cum_reward += reward
        self.cum_reward_hist.append(self._cum_reward)
        self.reward_history.append(reward)
