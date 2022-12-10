import argparse
import os
from typing import Tuple, Optional, Dict, Callable

import numpy as np
import torch

from meta_critics.agents.actor_critic.agent import Agent
from meta_critics.base_trainer.internal.call_interface import BaseCallbacks
from meta_critics.collectors.buffers.rollout_buffer import RolloutBuffer
from meta_critics.envs.util import create_env
from meta_critics.running_spec import RunningSpec


class EpisodeSampler:
    def __init__(self, envs, agent: Agent, spec: RunningSpec,
                 callback: Optional[list[Callable]] = None,
                 num_envs: Optional[int] = 4,
                 device="cpu"):
        """
        :param num_envs:
        :param device:
        """
        self._callbacks = BaseCallbacks(callbacks=callback)
        self._callbacks.register_trainer(self)

        self.gamma = spec.get("gamma")
        self.gae_lambda = spec.get("gamma_factor")

        self.is_gae = True
        assert self.gamma <= 1.0
        assert self.gae_lambda <= 1.0

        self.envs = envs
        self.num_envs = num_envs
        self.max_num_steps = 100
        self.rollout = RolloutBuffer(num_envs, self.max_num_steps,
                                     self.envs.single_observation_space.shape,
                                     self.envs.single_action_space.shape)

        self.global_step = 0
        self.best_score = 0
        self.device = device
        self.agent = agent

    def max_steps(self) -> int:
        return self.max_num_steps

    def get_num_envs(self) -> int:
        return self.num_envs

    def print_rollout(self):
        self.rollout.print_shapes()

    def hypothesis(self, next_obs: torch.Tensor, next_done: torch.Tensor) -> float:
        """
        Sample the sequence of actions, states and rewards an agent encounters
        over time is known as a trajectory.

        :param next_obs:
        :param next_done:
        :return: tensor next observation where observation for example for LunarLander
        Observation Shape 8 Observation High, Low
        """
        assert next_obs.shape[0] == self.num_envs
        assert next_done.shape[0] == self.num_envs

        total_reward = 0
        for step in range(0, self.max_num_steps):
            # action logic, take action update value for a step
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(next_obs)

            # take action , next observation and reward.
            next_obs, reward, done, truncated, info = self.envs.step(action.cpu().numpy())
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), \
                torch.Tensor(done).to(self.device)

            total_reward += reward.sum()
        return total_reward

    def sample_trajectory(self, next_obs: torch.Tensor,
                          next_done: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, Dict]:
        """
        Sample the sequence of actions, states and rewards an agent encounters
        over time is known as a trajectory.

        :param next_obs:
        :param next_done:
        :return: tensor next observation where observation for example for LunarLander
        Observation Shape 8 Observation High, Low
        """
        assert next_obs.shape[0] == self.num_envs
        assert next_done.shape[0] == self.num_envs

        done = next_done
        self._callbacks.on_episode_begin()

        for step in range(0, self.max_num_steps):
            self.global_step += 1 * self.num_envs
            self.rollout.obs[step] = next_obs
            self.rollout.dones[step] = next_done

            # action logic, take action update value for a step
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                self.rollout.values[step] = value.flatten()

            # update action step taken and logprob for given step.
            self.rollout.actions[step] = action
            self.rollout.logp[step] = logprob

            # take action , next observation and reward.
            next_obs, reward, done, truncated, info = self.envs.step(action.cpu().numpy())

            self.rollout.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), \
                torch.Tensor(done).to(self.device)

            self._callbacks.on_episode_end()

        return next_obs, reward, done, truncated, info

    def compute_gae_advantage(self, next_obs: torch.Tensor, next_done: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Take observation and done and compute and return advantage and returns.
        The advantage function, A, is the expected return of a state (value) subtracted
        from the expected return given an action in a state

        Where returns discounted sum of future returns

        :param next_obs:
        :param next_done:
        :return:
        """
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            gae_lambda = 0
            advantages = torch.zeros_like(self.rollout.rewards).to(self.device)
            for t in reversed(range(self.max_num_steps)):
                if t == self.max_num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - self.rollout.dones[t + 1]
                    next_values = self.rollout.values[t + 1]
                delta = self.rollout.rewards[t] + self.gamma * next_values * next_non_terminal - self.rollout.values[t]
                gae_lambda = delta + self.gamma * self.gae_lambda * next_non_terminal * gae_lambda
                advantages[t] = gae_lambda

            returns = advantages + self.rollout.values
            return returns, advantages

    def compute_advantage(self, next_obs: torch.Tensor, next_done: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param next_obs:
        :param next_done:
        :return:
        """
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            returns = torch.zeros_like(self.rollout.rewards).to(self.device)
            for t in reversed(range(self.max_num_steps)):
                if t == self.max_num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_return = next_value
                else:
                    next_non_terminal = 1.0 - self.rollout.dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = self.rollout.rewards[t] + self.gamma * next_non_terminal * next_return
            advantages = returns - self.rollout.values
            return returns, advantages

    def sample_batch(self, next_obs, next_done):
        """

        :param next_obs:
        :param next_done:
        :return:
        """
        trajectory_obs, _, _, _ = self.sample_trajectory(next_obs, next_done)
        if self.is_gae:
            returns, advantages = self.compute_gae_advantage(trajectory_obs, next_done)
        else:
            returns, advantages = self.compute_advantage(trajectory_obs, next_done)

        assert returns.shape[0] == self.max_num_steps
        assert advantages.shape[0] == self.max_num_steps
        assert returns.shape[1] == self.num_envs
        assert advantages.shape[1] == self.num_envs

        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_obs = self.rollout.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_actions = self.rollout.actions.reshape((-1,) + self.rollout.single_action_dim)
        b_logp = self.rollout.logp.reshape(-1)
        b_values = self.rollout.values.reshape(-1)

        return b_advantages, b_returns, b_obs, b_actions, b_logp, b_values

    # def sampler_last(self):

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return a batch of trajectories.  For example if num envs 2 and step size 128
        return shape [256, observation_space],  similarly advantages, return etc
        all [256]
        :return:
        """
        self.rollout.clean()
        initial_obs, initial_done = self.reset()
        return self.sample_batch(initial_obs, initial_done)

    def sample_to_completion(self):
        """

        :return:
        """
        episodes = []
        _next_obs, dones = self.reset()
        while not all(dones):
            self.global_step += 1 * self.num_envs
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(_next_obs)
            new_obs, reward, done, truncated, info = self.envs.step(action.cpu().numpy())
            reward = torch.tensor(reward).to(self.device).view(-1)
            episodes.append((torch.Tensor(_next_obs).to(self.device),
                             action,
                             torch.tensor(reward).to(self.device).view(-1),
                             logprob,
                             truncated))
            # update done and next observation
            _next_obs = new_obs
            dones = done

        return episodes

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: Tuple next state and resets all empty tensor all dones.
        """
        _next_obs, _infos = self.envs.reset()
        return torch.Tensor(_next_obs).to(self.device), \
            torch.zeros(self.num_envs).to(self.device)

    def close(self):
        self.envs.close()


def main():
    env_iterator = create_env(args.gym_id, args.seed, 1, capture_video=False)
    sampler = EpisodeSampler(env_iterator, num_envs=1)
    next_obs, next_done = sampler.reset()
    sampler.print_rollout()
    b_advantages, b_returns, b_obs, b_actions, b_logp, b_value = sampler.sample()

    print(b_advantages.shape)
    print(b_returns.shape)
    print("B observation shape", b_obs.shape)
    print(b_actions.shape)
    print(b_logp.shape)
    print(b_value.shape)

    sampler.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="LunarLander-v2",
                        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")

    parser.add_argument("--num-envs", type=int, default=4,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=500,
                        help="the number of steps to run in each environment per policy rollout")
    args = parser.parse_args()

    # env_iterator = create_env(args.gym_id, args.seed, args.num_envs, capture_video=False)
    # sampler = EpisodeSampler(env_iterator)
    # next_obs, next_done = sampler.reset()
    # print(next_obs.shape)
    # print(next_done.shape)
    # sampler.close()
    # single environment
