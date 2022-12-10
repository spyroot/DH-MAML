# Multi-armed bandit problems are a subset of MDPs where the agent’s environment is stateless.
# Specifically, there are k arms (actions), and at every time step, the agent pulls one of the arms, say
# i, and receives a reward drawn from an unknown distribution: our experiments take each arm to
# be a Bernoulli distribution with parameter pi.
# Mus
# mbayramo@stanford.edu
from typing import Optional
import numpy as np
from meta_critics.agents.bandits.base_bandit import MultiArmAgent
from meta_critics.envs.bandits.bandit_base_env import BanditEnv


class EpsGreedyMABAgent(MultiArmAgent):
    """
    Epsilon Greedy Action Selection Strategy.
    """

    def __init__(self, env_bandit: BanditEnv, eps: Optional[float] = 0.05):
        """

        param env_bandit: env_bandit: the environment of bandits agent need solve.
        param eps: a probability with which a random action is to be selected.
        """
        super(EpsGreedyMABAgent, self).__init__(env_bandit)
        self._quality = np.zeros(shape=(env_bandit.observation_space.shape[0],
                                        env_bandit.get_num_arms()))
        self._counts = np.zeros(shape=(env_bandit.observation_space.shape[0],
                                       env_bandit.get_num_arms()))
        self._eps = eps

    @property
    def eps(self) -> float:
        """Epsilon greedy probability.
        :return: float
        """
        return self._eps

    @property
    def quality(self) -> np.ndarray:
        """numpy.ndarray: Q values assigned by the policy to all actions"""
        return self._quality

    def select_action(self, context: int) -> int:
        """Selection an action according epsilon greedy strategy.
        Sutton and Barto chapter 2
        :param context:
        :return: action algorithm chosen
        """
        if np.random.random() < self.eps:
            action = np.random.randint(0, self._env_bandit.get_num_arms())
        else:
            action = np.argmax(self.quality[context])

        self.action_history.append((context, action))
        return action

    def update_params(self, context: int, action: int, reward: float) -> None:
        """Update parameters for the epsilon greedy policy.
        Updates the regret as the difference between max Q value and
        that of the action. Updates the Q values according to the
        reward received in this step
        :param context: context for which action is taken
        :param action: action agent taken for the step
        :param reward: reward obtained for the step
        """
        # update history and reward.
        self.reward_history.append(reward)
        self._regret += max(self.quality[context]) - self.quality[context, action]
        self.regret_history.append(self.regret)

        # update count
        reward_delta = (reward - self.quality[context, action])
        self.quality[context, action] += reward_delta / (self.counts[context, action] + 1)
        self.counts[context, action] += 1
