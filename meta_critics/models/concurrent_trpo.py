"""
The agent algorithm compute adaption ,
i.e grad in respect old policy, after we compute conjugate_gradient
and did a pass we compute surrogate_loss again.
"""
import traceback
from typing import Tuple

import torch
from torch import nn
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions import Distribution
from meta_critics.policies.policy import Policy
from meta_critics.running_spec import RunningSpec
from meta_critics.trajectory.advantage_episode import AdvantageBatchEpisodes
from meta_critics.policies.distribution_util import detach_dist_from_policy
from meta_critics.optimizers.optimization import conjugate_gradient
from meta_critics.base_trainer.torch_tools.torch_utils import weighted_mean
from meta_critics.models.async_gradient_learner import AsyncGradientBasedMetaLearner
from meta_critics.base_trainer.torch_tools.param_tools import vec2parameters


class ConcurrentMamlTRPO(AsyncGradientBasedMetaLearner):
    def __init__(self, policy: Policy, spec: RunningSpec):
        """
        Take policy that TRPO will use to compute KL terms and specs.
        Specs must contain CG line step parameters.

        The adapt method is inner loop optimization.
        Check the MAML paper for details.

        :param policy: policy based on reinforce
        :param spec:
        """
        super(ConcurrentMamlTRPO, self).__init__(policy, spec)
        assert self.policy is not None
        assert self.spec is not None

        self.ls_counter = 0
        self.fast_lr = self.spec.get('fast_lr', 'meta_task')
        self.first_order = self.spec.get('first_order', 'meta_task')

    def reinforce_loss(self, episodes: AdvantageBatchEpisodes, W=None) -> torch.Tensor:
        """Compute reinforce algorithm
        :param W:  weights of policy that algorithm will use to compute policy Pi
        :param episodes: current episode trajectory. 
        :return: Tensor of loss term.
        """
        try:
            episodes.to_gpu()
            pi = self.policy(episodes.observations.view((-1, *episodes.observation_shape)), W=W)
            log_probs = pi.log_prob(episodes.actions.view((-1, *episodes.action_shape)))
            log_probs = log_probs.view(torch.max(episodes.lengths), episodes.batch_size)
            losses = -weighted_mean(log_probs * episodes.advantages, lengths=episodes.lengths)
        except Exception as ex:
            # in case we have of bug it easy to track.
            print("actions      ## ", episodes.actions)
            print("observations ##", episodes.observations)
            print("episodes     ##", episodes.lengths)
            raise ex

        return losses.mean()

    def adapt(self, train_data, first_order=None):
        """Adaption based on maml algorithm
        :param train_data: A data used to compute inner loss
        :param first_order:  If we only perform first order. 
        :return:
        """
        if first_order is None:
            first_order = self.first_order

        theta = None
        inner_losses = []
        for i, futures in enumerate(train_data):
            inner_loss = self.reinforce_loss(futures, W=theta)
            theta = self.policy.update_params(inner_loss,
                                              params=theta,
                                              step_size=self.fast_lr,
                                              first_order=first_order)
            inner_losses.append(inner_loss)
        return theta, torch.mean(torch.stack(inner_losses))

    def surrogate_loss(self,
                       train_data,
                       valid_data,
                       detached_policy=None,
                       reduction='mean',
                       debug=True) -> Tuple[torch.Tensor, torch.Tensor, Distribution, torch.Tensor]:
        """
        TRPO surrogate loss
        :param debug:  mainly if we need dump debug data
        :param reduction: reduction mean or no reduction
        :param train_data: a batch of meta trask, trajectories per task
        :param valid_data: a batch of meta trask, trajectories per task
        :param detached_policy: a detach policy that surrogate_loss will use to compute KL term
        :return: Tensor of loss , kl term , detached policy i.e. old policy
        and inner_losses that we compute during adaption. It mainly useful for stats.
        """
        is_first_order = (detached_policy is not None) or self.first_order
        adapted_params, inner_losses = self.adapt(train_data, first_order=is_first_order)

        with torch.set_grad_enabled(detached_policy is None):
            valid_episodes = valid_data

            new_policy = self.policy(valid_episodes.observations, W=adapted_params)

            if detached_policy is None:
                detached_policy = detach_dist_from_policy(new_policy, self.device)

            # log ration
            valid_episodes.to_gpu()
            log_ratio = (new_policy.log_prob(valid_episodes.actions) - detached_policy.log_prob(valid_episodes.actions))
            ratio = torch.exp(log_ratio)
            loss_weighted = -weighted_mean(ratio * valid_episodes.advantages, lengths=valid_episodes.lengths)
            kls = weighted_mean(kl_divergence(new_policy, detached_policy), lengths=valid_episodes.lengths)

        if reduction == 'mean':
            return loss_weighted.mean(), kls.mean(), detached_policy, inner_losses
        return loss_weighted.mean(), kls.mean(), detached_policy, inner_losses

    async def step(self, train_trajec, valid_trajec, debug=True):
        """
        :param train_trajec: list of trajectories that forms an episode.
        :param valid_trajec: list of trajectories that forms an episode.
        :param debug:
        :return:
        """

        # need check why RPC bounce off to CPU sometime.
        self.policy.to(self.device)

        trpo_metrics = {}
        num_meta_tasks = len(train_trajec[0])
        data = list(zip(zip(*train_trajec), valid_trajec))

        # params = None
        # for i in range(0, len(data)):
        #     t, v = data[i]
        #     loss = self.reinforce_loss(t[0], W=params)
        #     params = self.policy.update_params(loss,
        #                                        params=params,
        #                                        step_size=0.2,
        #                                        first_order=True)

        old_losses = torch.empty(len(data), device=self.device)
        inner_loss = torch.empty(len(data), device=self.device)
        old_kl = torch.empty(len(data), device=self.device)
        old_policies = []

        for i in range(0, len(data)):
            t, v = data[i]
            old_losses[i], old_kl[i], old_policy, inner_loss[i] = self.surrogate_loss(t, v, detached_policy=None)
            old_policies.append(old_policy)

        old_losses = old_losses.sum() / num_meta_tasks
        inner_loss = inner_loss.sum() / num_meta_tasks
        old_kl = old_kl.sum() / num_meta_tasks

        trpo_metrics["inner_pre"] = inner_loss
        trpo_metrics["old_loss"] = old_losses
        trpo_metrics["old_kl"] = old_kl

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
            trpo_metrics['ls_step'] = 0
            for _ in range(self.ls_max_steps):
                vec2parameters(old_params - step_size * step, self.policy.parameters())

                _new_kl = torch.empty(len(data))
                _new_loss = torch.empty(len(data))
                _new_inner_loss = torch.empty(len(data))

                for i in range(0, len(data)):
                    t, v = data[i]
                    _new_loss[i], _new_kl[i], _, _new_inner_loss[i] = \
                        self.surrogate_loss(t, v, detached_policy=old_policies[i])

                new_improved_loss = (_new_loss.sum() / num_meta_tasks) - old_losses
                new_inner_loss = (_new_inner_loss.sum() / num_meta_tasks) - inner_loss
                new_kl = _new_kl.sum() / num_meta_tasks

                # line step
                trpo_metrics['ls_step'] += 1
                trpo_metrics['improved'] = new_improved_loss
                if (new_improved_loss.item() < 0.0) and (new_kl < _max_kl):
                    trpo_metrics['inner_post'] = new_inner_loss
                    trpo_metrics['loss_post'] = new_improved_loss
                    trpo_metrics['kl_post'] = new_kl
                    break

                step_size *= self.ls_backtrack_ratio
                self.ls_counter += 1
                trpo_metrics['ls_counter'] = self.ls_counter
            else:
                vec2parameters(old_params, self.policy.parameters())

        except Exception as trpo_err:
            print("TRPO Error: ", trpo_err)
            print(traceback.print_exc())
            raise trpo_err

        return trpo_metrics
