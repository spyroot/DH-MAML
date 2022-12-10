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
        num_tasks = len(train_futures[0])
        data = list(zip(zip(*train_futures), valid_futures))

        print(f"###### Num task {num_tasks}")
        print(f"len data {len(data)}")

        old_losses = []
        inner_loss = []
        old_kl = []
        for i in range(0, len(data)):
            t, v = data[i]
            outer_loss, kl_loss, old_policy, inner_adapt_loss = self.surrogate_loss(t, v, detached_policy=None)
            old_losses.append(outer_loss)
            old_kl.append(kl_loss)
            inner_loss.append(inner_adapt_loss)

        old_losses = torch.stack(old_losses).sum() / num_tasks
        inner_loss = torch.stack(inner_loss).sum() / num_tasks
        old_kl = torch.stack(old_kl).sum() / num_tasks

        print("old loss term", old_losses)
        logs["inner_pre"] = inner_loss
        logs["old_loss"] = old_losses
        logs["old_kl"] = old_kl

        try:

            old_kl = torch.sum(old_kl) / num_tasks
            old_loss = torch.sum(old_losses) / num_tasks
            grads = torch.autograd.grad(old_loss, list(self.policy.parameters()), retain_graph=True)
            grads = parameters_to_vector(grads)

            hvp = self.hessian_vector_product(old_kl)
            step_direction = conjugate_gradient(hvp, grads, cg_iters=self.cg_iters)
            shs = 0.5 * torch.dot(step_direction, hvp(step_direction, retain_graph=False))
            lagrange_multiplier = torch.sqrt(shs / self.max_kl)
            step = step_direction / lagrange_multiplier

            # old parameters
            old_params = parameters_to_vector(self.policy.parameters())
            step_size = 1.0
            self.max_kl = torch.tensor(self.max_kl)
            print("MAX_KL", self.max_kl)
            for _ in range(self.ls_max_steps):
                vec2parameters(old_params - step_size * step, self.policy.parameters())
                new_kl = []
                new_loss = []
                new_inner_loss = []
                for i in range(0, len(data)):
                    t, v = data[i]
                    _new_loss, _new_kl, _, _new_inner = self.surrogate_loss(t, v, detached_policy=old_policy)
                    new_kl.append(_new_kl)
                    new_loss.append(_new_loss)
                    new_inner_loss.append(_new_inner)

                new_improved_loss = (torch.stack(new_loss).sum() / num_tasks) - old_loss
                new_inner_loss = (torch.stack(new_inner_loss).sum() / num_tasks) - inner_loss
                new_kl = torch.stack(new_kl).sum() / num_tasks

                logs['improved'] = new_improved_loss

                print(new_improved_loss.item() < 0.0)
                print(new_kl < self.max_kl)

                if (new_improved_loss.item() < 0.0) and (new_kl < self.max_kl):
                    print("### Best loss ", new_improved_loss)
                    print("kl_post", new_kl)
                    logs['inner_post'] = new_inner_loss
                    logs['loss_post'] = new_improved_loss
                    logs['kl_post'] = new_kl
                    break
                else:
                    print("############# current improved "
                          "loss {:.4} kl {:.4} max {:.4}".format(new_improved_loss.item(),
                                                                 new_kl.item(), self.max_kl.item()))
                # else:
                # print("improved")
                step_size *= self.ls_backtrack_ratio
            else:
                vec2parameters(old_params, self.policy.parameters())

        except Exception as trpo_err:
            print("TRPO Error: ", trpo_err)
            raise trpo_err

        return logs

    # async def step(self, train_futures, valid_futures, debug=True):
    #     """
    #
    #     :param train_futures:
    #     :param valid_futures:
    #     :param debug:
    #     :return:
    #     """
    #
    #     logs = {}
    #     num_tasks = len(train_futures[0])
    #
    #     for i, (train, valid) in enumerate(zip(zip(*train_futures), valid_futures)):
    #         old_losses, old_kl, old_pis, inner_loss = self.surrogate_loss(train, valid, detached_policy=None)
    #         logs["old_losses"] = old_losses
    #         logs["old_kl"] = old_kl
    #
    #         # print(f"i old_losses {old_losses}")
    #         # print(f"i old_losses {old_kl}")
    #         # print(f"i old_losses {old_pis}")
    #         # print(f"i old_losses {inner_loss}")
    #
    #     grads = torch.autograd.grad(old_losses, list(self.policy.parameters()), retain_graph=True)
    #     grads = parameters_to_vector(grads)
    #
    #     hessian_vector_product = self.hessian_vector_product(old_kl)
    #     step_direction = conjugate_gradient(hessian_vector_product, grads, cg_iters=self.cg_iters)
    #     shs = 0.5 * torch.dot(step_direction, hessian_vector_product(step_direction, retain_graph=False))
    #     lagrange_multiplier = torch.sqrt(shs / self.max_kl)
    #     step = step_direction / lagrange_multiplier
    #
    #     # old parameters
    #     old_params = parameters_to_vector(self.policy.parameters())
    #     step_size = 1.0
    #     for _ in range(self.ls_max_steps):
    #         vec2parameters(old_params - step_size * step, self.policy.parameters())
    #         for (train, valid) in zip(zip(*train_futures), valid_futures):
    #             losses, kl, _, inner_loss = self.surrogate_loss(train, valid, detached_policy=old_pis)
    #
    #         improve = losses - old_kl
    #         logs['improved'] = improve
    #
    #         if (improve.item() < 0.0) and (kl.item() < self.max_kl):
    #             logs['loss_post'] = losses
    #             logs['inner_post'] = inner_loss
    #             logs['kl_post'] = kl
    #             # logs['improved'] = improve
    #             break
    #         step_size *= self.ls_backtrack_ratio
    #     else:
    #         vec2parameters(old_params, self.policy.parameters())
    #
    #     return logs

#
# from typing import Tuple
#
# import torch
# from torch import nn
# from torch.distributions.kl import kl_divergence
# from torch.nn.utils.convert_parameters import parameters_to_vector
#
# from NamedEpisode import NamedEpisode
# from meta_critics.maml_rl.metalearners.async_gradient_learner import AsyncGradientBasedMetaLearner
# from meta_critics.policies.policy import Policy
# from meta_critics.optimizers.optimization import conjugate_gradient
# from meta_critics.objective.reinforcement_learning import reinforce_loss
# from meta_critics.running_spec import RunningSpec
# from meta_critics.base_trainer.torch_tools.param_tools import vec2parameters
# from meta_critics.base_trainer.torch_tools.torch_utils import (weighted_mean, detach_distribution)
#
#
# class ConcurrentMamlTRPO(AsyncGradientBasedMetaLearner):
#     def __init__(self, policy: Policy, spec: RunningSpec):
#         """
#
#         :param policy:
#         :param spec:
#         """
#         super(ConcurrentMamlTRPO, self).__init__(policy, spec)
#         assert self.policy is not None
#         assert self.spec is not None
#
#         self.fast_lr = self.spec.get('fast_lr', 'meta_task')
#         self.first_order = self.spec.get('first_order', 'meta_task')
#
#     def reinforce_loss(self, episodes: NamedEpisode, W=None) -> torch.Tensor:
#         """
#         :param W:
#         :param policy:
#         :param episodes:
#         :return:
#         """
#
#         # print("reinforce_loss episodes.observations", episodes.observations.shape)
#         # print("reinforce_loss episodes.actions", episodes.actions.shape)
#         # t = episodes.observation_shape
#         # val = episodes.observations.view((-1, *episodes.observation_shape)).float()
#         # pi = policy(val, W=W)
#         try:
#             # print("episodes obs", episodes.observations)
#             # print("episodes act", episodes.actions)
#             # print("episodes act", episodes.lengths)
#             print(f"Computing reinforce_loss type{type(episodes)}")
#             pi = self.policy(episodes.observations.view((-1, *episodes.observation_shape)).float(), W=W)
#             # episodes.require_grad()
#             # episodes.to_gpu()
#             log_probs = pi.log_prob(episodes.actions.view((-1, *episodes.action_shape)))
#             log_probs = log_probs.view(torch.max(episodes.lengths), episodes.batch_size)
#             losses = -weighted_mean(log_probs * episodes.advantages, lengths=episodes.lengths)
#         except Exception as ex:
#             print("ac ## ", episodes.actions)
#             print("ob ##", episodes.observations)
#             print("lt ##", episodes.lengths)
#             raise ex
#
#         return losses.mean()
#
#     async def adapt(self, train_futures, first_order=None):
#         """
#         :param train_futures:
#         :param first_order:
#         :return:
#         """
#         print("Computing adapt")
#
#         if first_order is None:
#             first_order = self.first_order
#
#         # Loop over the number of steps of adaptation
#         theta = None
#         inner_losses = []
#         for futures in train_futures:
#             inner_loss = self.reinforce_loss(await futures, W=theta)
#             # inner_loss = reinforce_loss(self.policy, await futures, W=theta)
#             print("Adap updating policy")
#             theta = self.policy.update_params(inner_loss, params=theta,
#                                               step_size=self.fast_lr,
#                                               first_order=first_order)
#             inner_losses.append(inner_loss)
#         return theta, torch.mean(torch.stack(inner_losses))
#
#     async def surrogate_loss(self, train_futures, valid_futures, old_policy=None,
#                              reduction='mean', debug=True) -> Tuple[torch.Tensor,
#                                                                     torch.Tensor,
#                                                                     nn.Module, torch.Tensor]:
#         """
#
#         :param debug:
#         :param reduction:
#         :param train_futures:
#         :param valid_futures:
#         :param old_policy:
#         :return:
#         """
#         print("Computing surrogate_loss")
#         first_order = (old_policy is not None) or self.first_order
#         adapted_params, inner_losses = await self.adapt(train_futures, first_order=first_order)
#
#         with torch.set_grad_enabled(old_policy is None):
#             valid_episodes = await valid_futures
#             new_policy = self.policy(valid_episodes.observations.float(), W=adapted_params)
#
#             if old_policy is None:
#                 old_policy = detach_distribution(new_policy)
#
#             # log ration
#             valid_episodes.to_gpu()
#             log_ratio = (new_policy.log_prob(valid_episodes.actions) - old_policy.log_prob(valid_episodes.actions))
#             ratio = torch.exp(log_ratio)
#             losses = -weighted_mean(ratio * valid_episodes.advantages, lengths=valid_episodes.lengths)
#             kls = weighted_mean(kl_divergence(new_policy, old_policy), lengths=valid_episodes.lengths)
#
#         print(f"surrogate_loss mean {losses.mean()}")
#
#         return losses.mean(), kls.mean(), old_policy, inner_losses
#
#     def step(self, train_futures, valid_futures, debug=True):
#         """
#
#         :param train_futures:
#         :param valid_futures:
#         :param debug:
#         :return:
#         """
#
#         logs = {}
#         num_tasks = len(train_futures[0])
#
#         # Compute the surrogate loss
#         old_losses, old_kls, old_pis, inner_loss = self._async_gather(
#                 [self.surrogate_loss(train, valid, old_policy=None)
#                  for (train, valid) in
#                  zip(zip(*train_futures), valid_futures)])
#
#         logs['loss_pre'] = old_losses
#         logs['inner_pre'] = inner_loss
#         logs['kl_pre'] = old_kls
#
#         # for l in inner_loss:
#         #     print(type(l))
#         #     print(l.__class__)
#         #     print("inner_loss loss grad", l.grad)
#         #
#         # for l in old_losses:
#         #     print(type(l))
#         #     print(l.__class__)
#         #     print("inner_loss loss grad", l.grad)
#         #
#         # for l in old_kls:
#         #     print(type(l))
#         #     print(l.__class__)
#         #     print("inner_loss loss grad", l.grad)
#         #
#         # for l in old_pis:
#         #     print(type(l))
#         #     print(l.__class__)
#         #     print("inner_loss loss grad", l.grad)
#         #
#         old_loss = sum(old_losses) / num_tasks
#         # print("Old loss grad", old_losses.grad)
#         # print("inner_loss loss grad", inner_loss.grad)
#
#     #    optimizer = AdaHessian(self.policy.parameters())
#         print("Computing grad")
#
#         grads = torch.autograd.grad(old_loss, list(self.policy.parameters()), retain_graph=True)
#         grads = parameters_to_vector(grads)
#         #  optimizer.zero_grad()
#         # old_loss.backward(create_graph=True, retain_graph=True)
#         # optimizer.step()
#
#         # Compute the step direction with Conjugate Gradient
#         # Compute the Lagrange multiplier
#         old_kl = sum(old_kls) / num_tasks
#         hessian_vector_product = self.hessian_vector_product(old_kl)
#         step_direction = conjugate_gradient(hessian_vector_product, grads, cg_iters=self.cg_iters)
#         shs = 0.5 * torch.dot(step_direction, hessian_vector_product(step_direction, retain_graph=False))
#         lagrange_multiplier = torch.sqrt(shs / self.max_kl)
#         step = step_direction / lagrange_multiplier
#
#         # old parameters
#         old_params = parameters_to_vector(self.policy.parameters())
#         step_size = 1.0
#         for _ in range(self.ls_max_steps):
#             vec2parameters(old_params - step_size * step, self.policy.parameters())
#             losses, kls, _, inner_loss = self._async_gather([
#                 self.surrogate_loss(train, valid, old_policy=old_pi)
#                 for (train, valid, old_pi)
#                 in zip(zip(*train_futures), valid_futures, old_pis)])
#
#             improve = (sum(losses) / num_tasks) - old_loss
#             logs['improved'] = improve
#
#             kl = sum(kls) / num_tasks
#             if (improve.item() < 0.0) and (kl.item() < self.max_kl):
#                 logs['loss_post'] = losses
#                 logs['inner_post'] = inner_loss
#                 logs['kl_post'] = kls
#                 # logs['improved'] = improve
#                 break
#             step_size *= self.ls_backtrack_ratio
#         else:
#             vec2parameters(old_params, self.policy.parameters())
#
#         return logs
