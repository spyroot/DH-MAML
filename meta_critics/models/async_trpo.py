import traceback
from typing import Tuple

import torch
from torch import nn
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import parameters_to_vector

from meta_critics.base_trainer.torch_tools.param_tools import vec2parameters
from meta_critics.base_trainer.torch_tools.torch_utils import weighted_mean
from meta_critics.models.async_gradient_learner import AsyncGradientBasedMetaLearner
from meta_critics.optimizers.optimization import conjugate_gradient
from meta_critics.policies.distribution_util import detach_dist_from_policy
from meta_critics.policies.policy import Policy
from meta_critics.running_spec import RunningSpec
from meta_critics.trajectory.advantage_episode import AdvantageBatchEpisodes


class ConcurrentMamlTRPO(AsyncGradientBasedMetaLearner):
    def __init__(self, policy: Policy, spec: RunningSpec):
        """

        :param policy:
        :param spec:
        """
        super(ConcurrentMamlTRPO, self).__init__(policy, spec)
        assert self.policy is not None
        assert self.spec is not None

        self.ls_counter = 0
        self.fast_lr = self.spec.get('fast_lr', 'meta_task')
        self.first_order = self.spec.get('first_order', 'meta_task')

    def reinforce_loss(self, episodes: AdvantageBatchEpisodes, W=None) -> torch.Tensor:
        """
        :param W:
        :param episodes:
        :return:
        """
        try:
            episodes.to_gpu()
            pi = self.policy(episodes.observations.view((-1, *episodes.observation_shape)).float(), W=W)
            log_probs = pi.log_prob(episodes.actions.view((-1, *episodes.action_shape)))
            log_probs = log_probs.view(torch.max(episodes.lengths), episodes.batch_size)
            losses = -weighted_mean(log_probs * episodes.advantages, lengths=episodes.lengths)
        except Exception as ex:
            print("ac ## ", episodes.actions)
            print("ob ##", episodes.observations)
            print("lt ##", episodes.lengths)
            raise ex

        return losses.mean()

    def adapt(self, train_futures, first_order=None):
        """
        :param train_futures:
        :param first_order:
        :return:
        """

        if first_order is None:
            first_order = self.first_order

        theta = None
        inner_losses = []
        for i, futures in enumerate(train_futures):
            inner_loss = self.reinforce_loss(futures, W=theta)
            theta = self.policy.update_params(inner_loss,
                                              params=theta,
                                              step_size=self.fast_lr,
                                              first_order=first_order)
            inner_losses.append(inner_loss)
        return theta, torch.mean(torch.stack(inner_losses))

    def surrogate_loss(self,
                       train_futures,
                       valid_futures,
                       detached_policy=None,
                       reduction='mean',
                       debug=True) -> Tuple[torch.Tensor, torch.Tensor, nn.Module, torch.Tensor]:
        """

        :param debug:
        :param reduction:
        :param train_futures:
        :param valid_futures:
        :param detached_policy:
        :return:
        """
        is_first_order = (detached_policy is not None) or self.first_order
        adapted_params, inner_losses = self.adapt(train_futures, first_order=is_first_order)

        with torch.set_grad_enabled(detached_policy is None):
            valid_episodes = valid_futures
            new_policy = self.policy(valid_episodes.observations.float(), W=adapted_params)

            if detached_policy is None:
                detached_policy = detach_dist_from_policy(new_policy, self.device)

            # log ration
            valid_episodes.to_gpu()
            log_ratio = (new_policy.log_prob(valid_episodes.actions) - detached_policy.log_prob(valid_episodes.actions))
            ratio = torch.exp(log_ratio)
            losses = -weighted_mean(ratio * valid_episodes.advantages, lengths=valid_episodes.lengths)
            kls = weighted_mean(kl_divergence(new_policy, detached_policy), lengths=valid_episodes.lengths)

        return losses.mean(), kls.mean(), detached_policy, inner_losses

    async def step(self, train_futures, valid_futures, debug=True):
        """

        :param train_futures:
        :param valid_futures:
        :param debug:
        :return:
        """

        logs = {}
        num_meta_tasks = len(train_futures[0])
        data = list(zip(zip(*train_futures), valid_futures))

        old_losses = torch.empty(len(data))
        inner_loss = torch.empty(len(data))
        old_kl = torch.empty(len(data))
        old_policies = []

        for i in range(0, len(data)):
            t, v = data[i]
            old_losses[i], old_kl[i], old_policy, inner_loss[i] = self.surrogate_loss(t, v, detached_policy=None)
            old_policies.append(old_policy)

        old_losses = old_losses.sum() / num_meta_tasks
        inner_loss = inner_loss.sum() / num_meta_tasks
        old_kl = old_kl.sum() / num_meta_tasks

        logs["inner_pre"] = inner_loss
        logs["old_loss"] = old_losses
        logs["old_kl"] = old_kl

        try:

            grads = torch.autograd.grad(old_losses, list(self.policy.parameters()), retain_graph=True)
            grads = parameters_to_vector(grads)

            hvp = self.hessian_vector_product(old_kl)
            step_direction = conjugate_gradient(hvp, grads, cg_iters=self.cg_iters)
            shs = 0.5 * torch.dot(step_direction, hvp(step_direction, retain_graph=False))
            lagrange_multiplier = torch.sqrt(shs / self.max_kl)
            step = step_direction / lagrange_multiplier

            # old parameters
            old_params = parameters_to_vector(self.policy.parameters())
            step_size = 1.0

            _max_kl = torch.tensor(self.max_kl)
            logs['ls_step'] = 0
            for _ in range(self.ls_max_steps):
                vec2parameters(old_params - step_size * step, self.policy.parameters())

                _new_kl = torch.empty(len(data))
                _new_loss = torch.empty(len(data))
                _new_inner_loss = torch.empty(len(data))

                for i in range(0, len(data)):
                    t, v = data[i]
                    _new_loss[i], _new_kl[i], _, _new_inner_loss[i] = self.surrogate_loss(t, v,
                                                                                       detached_policy=old_policies[i])

                new_improved_loss = (_new_loss.sum() / num_meta_tasks) - old_losses
                new_inner_loss = (_new_inner_loss.sum() / num_meta_tasks) - inner_loss
                new_kl = _new_kl.sum() / num_meta_tasks

                logs['ls_step'] += 1
                logs['improved'] = new_improved_loss
                if (new_improved_loss.item() < 0.0) and (new_kl < _max_kl):
                    logs['inner_post'] = new_inner_loss
                    logs['loss_post'] = new_improved_loss
                    logs['kl_post'] = new_kl
                    break

                step_size *= self.ls_backtrack_ratio
                self.ls_counter += 1
                logs['ls_counter'] = self.ls_counter
            else:
                vec2parameters(old_params, self.policy.parameters())

        except Exception as trpo_err:
            print("TRPO Error: ", trpo_err)
            print(traceback.print_exc())
            raise trpo_err

        return logs
