## This need to be moved to a new trainer..
import argparse
import os
import time
from abc import ABC
from typing import Optional, Callable

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from meta_critics.agents.actor_critic.agent import Agent
from meta_critics.base_trainer.env_creator import SimCreator
from meta_critics.collectors.episode_sampler import EpisodeSampler
from meta_critics.base_trainer.internal.base_trainer import BaseTrainer
from meta_critics.base_trainer.internal.call_interface import BaseCallbacks
from meta_critics.envs.util import create_env


class Spec:
    def __init__(self):
        # the surrogate clipping coefficient
        # https://arxiv.org/abs/2202.00079
        # coefficient of the entropy
        # the maximum norm for the gradient clipping
        # normalize advantages
        # weather to clip loss for value function
        # num env run
        # max step in trajectory
        # run name
        # size in mini batch

        self.spec = {'vf_coef': 0.5,
                     'clip_coef': 0.2,
                     'update_epochs': 4,
                     'ent_coef': 0.01,
                     'max_grad_norm': 0.5,
                     'norm_adv': True,
                     'clip_value_fn_loss': True,
                     'num_envs': 4,
                     'max_num_steps': 128,
                     'run_name': "test",
                     'mini_batch_size': 4,
                     'learning-rate': 2.5e-4,
                     'total_timesteps': 250000
                     }

    def vf_coef(self):
        return self.spec['vf_coef']

    def max_grad_norm(self):
        return self.spec['max_grad_norm']

    def ent_coef(self):
        return self.spec['ent_coef']

    def update_epochs(self):
        return self.spec['update_epochs']

    # def num_envs(self):
    #     return self.spec['num_envs']

    def max_num_steps(self):
        return self.spec['max_num_steps']

    def mini_batch_size(self):
        return self.spec['mini_batch_size']

    def clip_coef(self):
        return self.spec['clip_coef']

    # def learning_rate(self):
    #     return self.spec['learning-rate']

    def total_timesteps(self):
        return self.spec['learning-rate']

    def update(self, k, v):
        self.spec[k] = v


class Trainer(BaseTrainer, ABC):
    def __init__(self,
                 sample: EpisodeSampler,
                 agent: Agent,
                 spec: Spec,
                 callback: Optional[list[Callable]] = None,
                 writer: Optional[SummaryWriter] = None,
                 device="cpu", debug=False):
        super(Trainer, self).__init__(device=device)
        """
        :param num_envs:
        :param max_num_steps:
        :param num_mini-batches:
        :param device:
        """

        self._callbacks = BaseCallbacks(callbacks=callback)
        self._callbacks.register_trainer(self)
        # self._callbacks.register_metric(self.metric)

        # coefficient of the value function
        self.best_score = float('-inf')
        self.vf_coef = spec.vf_coef()
        # the surrogate clipping coefficient
        # https://arxiv.org/abs/2202.00079
        self.clip_coef = spec.clip_coef()
        # epoch num policy update
        self.update_epochs = spec.update_epochs()
        # coefficient of the entropy
        self.ent_coef = spec.ent_coef()
        # the maximum norm for the gradient clipping
        self.max_grad_norm = spec.max_grad_norm()
        # normalize advantages
        self.norm_adv = True
        # weather to clip loss for value function
        self.clip_value_fn_loss = True

        self.debug = debug
        self.num_envs = int(spec.num_envs)
        self.max_num_steps = spec.max_num_steps()
        self.mini_batch_size = spec.mini_batch_size()

        # discount gamma
        self.device = device
        self.total_timesteps = spec.total_timesteps()
        self.batch_size = int(self.num_envs * self.max_num_steps)
        self.minibatch_size = int(self.batch_size // self.mini_batch_size)
        self.num_updates = self.total_timesteps // self.batch_size
        self.learning_rate = float(spec.learning_rate)
        self.sampler = sample
        self.agent = agent

        self.optimizer = optim.Adam(self.agent.parameters(),
                                    lr=self.learning_rate, eps=1e-5)
        self.global_step = 0
        self.writer = writer

    def anneal_if_needed(self, update):
        frac = 1.0 - (update - 1.0) / self.num_updates
        self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

    def save(self):
        pass

    def load(self):
        pass

    def clip_value_loss(self, minibatch_new_values: torch.Tensor,
                        minibatch_returns: torch.Tensor,
                        minibatch_values: torch.Tensor) -> torch.Tensor:
        """
        :param minibatch_values:
        :param minibatch_returns:
        :param minibatch_new_values:
        :return:
        """
        v_loss_unclipped = (minibatch_new_values - minibatch_returns) ** 2
        clipped = minibatch_values + torch.clamp(minibatch_new_values - minibatch_values,
                                                 -self.clip_coef, self.clip_coef)
        v_loss_clipped = (clipped - minibatch_returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
        return v_loss

    def train(self, ls_max_steps=2):
        """
        :return:
        """
        self._train()
        # self.envs.close()

    def evaluate(self):
        """
        :return:
        """
        _next_obs, done = self.sampler.reset()
        total_reward = self.sampler.hypothesis(_next_obs, done)

    def _train(self):
        """
        :return:
        """
        next_obs, next_done = self.sampler.reset()
        tqdm_iter = tqdm(range(1, self.num_updates + 1),
                         desc=f"Training in progress, device {self.device}")

        for update in tqdm_iter:
            self.anneal_if_needed(update)
            loss, metrics = self._train_step(next_obs, next_done)
            tqdm_update_dict = \
                {'loss': loss.item()}
            tqdm_iter.set_postfix(tqdm_update_dict)

            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.state.step)
            self.writer.add_scalar("losses/value_loss", metrics['value_loss'], self.state.step)
            self.writer.add_scalar("losses/policy_loss", metrics['policy_loss'], self.state.step)
            self.writer.add_scalar("losses/entropy", metrics['entropy_loss'], self.state.step)
            self.writer.add_scalar("losses/old_approx_kl", metrics['old_approx_kl'], self.state.step)
            self.writer.add_scalar("losses/approx_kl", metrics['approx_kl'], self.state.step)
            # self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.state.step)
            self.writer.add_scalar("losses/variance", metrics['variance'], self.state.step)
            self.writer.add_scalar("charts/sps", metrics['sps'], self.state.step)

    def _train_step(self, next_obs, next_done):
        """
        :return:
        """
        start_time = time.time()
        # next_obs, next_done = self.sampler.reset()

        metrics = {}
        clipfracs = []
        all_losses = torch.zeros(self.update_epochs * self.num_envs)

        b_advantages, b_returns, b_obs, b_actions, b_logp, b_values = self.sampler.sample_batch(next_obs, next_done)
        self.state.step += 1 * self.num_envs

        batch_indices = torch.arange(self.batch_size)
        batch_indices.to(self.device)

        for j, n_updates in enumerate(range(self.update_epochs)):
            # print("Start", n_updates)
            # premute batch
            idx = torch.randperm(batch_indices.nelement())
            batch_indices = batch_indices.view(-1)[idx].view(batch_indices.size())
            # start is offset with step size.
            for start in range(0, self.batch_size, self.minibatch_size):
                minibatch_indices = batch_indices[start:start + self.minibatch_size]
                assert minibatch_indices.shape[0] == self.minibatch_size
                minibatch_advantaged = b_advantages[minibatch_indices]
                minibatch_observation = b_obs[minibatch_indices]
                minibatch_action = b_actions.long()[minibatch_indices]
                minibatch_logp = b_logp[minibatch_indices]
                minibatch_returns = b_returns[minibatch_indices]
                minibatch_values = b_values[minibatch_indices]

                _, new_logp, entropy, new_value = self.agent.get_action_and_value(minibatch_observation,
                                                                                  minibatch_action)
                assert new_logp.shape == minibatch_logp.shape
                # ration between log probs
                log_difference = new_logp - minibatch_logp
                log_ratio_exp = (new_logp - minibatch_logp).exp()
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-log_difference).mean()
                    approx_kl = ((log_ratio_exp - 1) - log_difference).mean()
                    clipfracs += [((log_ratio_exp - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = minibatch_advantaged
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * log_ratio_exp
                pg_loss2 = -mb_advantages * torch.clamp(log_ratio_exp, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                new_value = new_value.view(-1)
                if self.clip_value_fn_loss:
                    v_loss = self.clip_value_loss(new_value, minibatch_returns, minibatch_values)
                else:
                    v_loss = 0.5 * ((new_value - minibatch_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                all_losses[j] = loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics['value_loss'] = v_loss.item()
                metrics['entropy_loss'] = entropy_loss.item()
                metrics['policy_loss'] = pg_loss.item()
                metrics['approx_kl'] = approx_kl.item()
                metrics['old_approx_kl'] = old_approx_kl.item()

            if self.debug:
                break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)

            # self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.state.step)
            # self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.state.step)
            metrics['sps'] = int(self.state.step / (time.time() - start_time))
            metrics['variance'] = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return all_losses.mean(), metrics


class Simulation:
    def __init__(self, cmd):
        self._is_evaluation = False
        self._is_record = False

        self.exp_name = self.experiment_name(cmd)
        self.writer = SummaryWriter(f"runs/{self.exp_name}")
        self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(cmd).items()])),
        )

    def experiment_name(self, cmd):
        return f'{cmd.exp_name}' \
               f'.support_{cmd.num_envs}' \
               f'.seed_{cmd.seed}' \
               f'outer_lr_{cmd.learning_rate}.' \
               f'num_steps_{cmd.num_steps}'

    def is_record(self, step: int):
        return self._is_record

    def set_record(self, val: bool):
        self._is_evaluation = val

    def is_evaluation(self):
        return self._is_evaluation

    def set_evaluation(self, val: bool):
        self._is_evaluation = val

    def evaluate(self, trainer):
        self.set_evaluation(True)
        trainer.evaluate()
        self.set_evaluation(False)

    def train(self, trainer):
        self.set_evaluation(False)
        trainer.evaluate()
        self.set_evaluation(False)

    def run(self, trainer_spec):
        """
        :param trainer_spec:
        :return:
        """
        env_iterator = create_env(trainer_spec.gym_id,
                                  trainer_spec.seed,
                                  trainer_spec.num_envs,
                                  run_name=self.exp_name,
                                  episode_rec_trigger=self.is_record,
                                  episode_log_trigger=self.is_evaluation,
                                  capture_video=True)
        sim_creator = SimCreator(env_iterator)
        envs = sim_creator.get_env()

        agent = Agent(envs)
        sampler = EpisodeSampler(envs, agent, num_envs=trainer_spec.num_envs)
        trainer = Trainer(sampler, agent, trainer_spec, writer=self.writer, debug=False)
        self.evaluate(trainer)

        # trainer.train()
        # self.set_evaluation(True)
        # trainer.evaluate()


