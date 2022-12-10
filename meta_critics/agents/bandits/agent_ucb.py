from typing import Optional

import numpy as np
from meta_critics.agents.bandits.base_bandit import MultiArmAgent
from meta_critics.envs.bandits.bandit_base_env import BanditEnv


class UCBAgent(MultiArmAgent):
    """
    Multi-Armed Bandit Solver with Upper Confidence Bound based

    Asymptotically efficient adaptive allocation rules,
    Lai and Robbins
    """
    def __init__(self, env_bandit: BanditEnv, confidence: Optional[float] = 1.0):
        super(UCBAgent, self).__init__(env_bandit)
        self._c = confidence
        self._quality = np.zeros(shape=(env_bandit.observation_space.shape[0], env_bandit.get_num_arms()))
        self._counts = np.zeros(shape=(env_bandit.observation_space.shape[0], env_bandit.get_num_arms()))
        self._t = 0

    @property
    def confidence(self) -> float:
        """float: Confidence level which weights the exploration term"""
        return self._c

    @property
    def quality(self) -> np.ndarray:
        """numpy.ndarray: q values assigned by the policy to all actions"""
        return self._quality

    def select_action(self, context: int) -> int:
        """Select an action according to UCB algorithm.
        Take action that maximises a weighted sum of the Q values for the action
        and an exploration encouragement term controlled by confidence
        :param context: the context to select action for
        :returns: Selected action
        """
        self._t += 1
        ucb_criterion = self.quality[context] + self.confidence * np.sqrt(2 * np.log(self._t + 1) / (self.counts[context] + 1))
        action = np.argmax(ucb_criterion)
        self.action_history.append((context, action))
        return action

    def update_params(self, context: int, action: int, reward: float) -> None:
        """

        :param context:
        :param action:
        :param reward:
        :return:
        """
        self.reward_history.append(reward)
        self._regret += max(self.quality[context]) - self.quality[context, action]
        self.reward_history.append(self.regret)
        self.quality[context, action] += (reward - self.quality[context, action]) / (self.counts[context, action] + 1)
        self.counts[context, action] += 1
