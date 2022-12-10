from datetime import datetime
from typing import Optional, Any
import numpy as np
from meta_critics.agents.trainer.bandit_base_trainer import BanditTrainer
from meta_critics.envs.bandits.bandit_base_env import BanditEnv


class MultiArmBanditTrainer(BanditTrainer):
    def __init__(self, agent: Any, bandit_env: BanditEnv, seed: int = 1,
                 time_steps: Optional[int] = 10,
                 log_freq: Optional[int] = 1):
        """

        :param agent:
        :param bandit_env:
        :param seed:
        :param time_steps:
        :param log_freq:
        """
        super(MultiArmBanditTrainer, self).__init__(agent, bandit_env)
        self.log_freq = log_freq
        self._seed = seed
        self._time_steps = time_steps
        self._context = 0

    def progress_callback(self, step: int, reward, regret_moving_avg, reward_moving_avg):
        print(
                # self.logger.write(
                {
                    "timestep": step,
                    "regret/regret": self.agent.regret_history[-1],
                    "reward/reward": reward,
                    "regret/cumulative_regret": self.agent.cum_regret,
                    "reward/cumulative_reward": self.agent.cum_reward,
                    "regret/regret_moving_avg": regret_moving_avg[-1],
                    "reward/reward_moving_avg": reward_moving_avg[-1],
                }
        )

    def train(self) -> Any:
        """
        Train agent
        :return: return all metrics in dict
        """

        start_time = datetime.now()
        print(
                f"\nStarted at {start_time:%d-%m-%y %H:%M:%S}\n"
                f"Training {self.agent.__class__.__name__} on {self.bandit_env.__class__.__name__} "
                f"for {self._time_steps} timesteps, seed={self._seed}"
        )

        mv_len = self._time_steps // 20
        _, _ = self.bandit_env.reset(seed=self._seed)
        regret_moving_avg = []
        reward_moving_avg = []

        for t in range(1, self._time_steps + 1):
            action = self.agent.select_action(self._context)
            _, reward, terminated, truncated, info = self.bandit_env.step(action)
            self.agent.receive_reward(reward)
            self.agent.action_history.append(action)
            self.agent.update_params(self._context, action, reward)

            regret_moving_avg.append(np.mean(self.agent.regret_history[-mv_len:]))
            reward_moving_avg.append(np.mean(self.agent.reward_history[-mv_len:]))

            if self.log_freq > 0 and t % self.log_freq == 0:
                self.progress_callback(t, reward, reward_moving_avg, reward_moving_avg)
        print(
                f"Training completed in {(datetime.now() - start_time).seconds} seconds\n"
                f"Final Regret Moving Average: {regret_moving_avg[-1]} | "
                f"Final Reward Moving Average: {reward_moving_avg[-1]}"
        )

        return {
            "regrets": self.agent.regret_history,
            "rewards": self.agent.reward_history,
            "cumulative_regrets": self.agent.cum_regret_hist,
            "cumulative_rewards": self.agent.cum_reward_hist,
            "regret_moving_avgs": regret_moving_avg,
            "reward_moving_avgs": reward_moving_avg,
        }
