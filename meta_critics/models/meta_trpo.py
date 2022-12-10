import torch
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import parameters_to_vector

from meta_critics.policies.distribution_util import detach_dist_from_policy
from meta_critics.policies.policy import Policy
from meta_critics.optimizers.optimization import conjugate_gradient
from meta_critics.objective.reinforcement_learning import reinforce_loss
from meta_critics.base_trainer.torch_tools.torch_utils import (weighted_mean)
from meta_critics.running_spec import RunningSpec
from meta_critics.base_trainer.torch_tools.param_tools import vec2parameters
from meta_critics.base_trainer.torch_tools.tensor_tools import to_numpy
from meta_critics.models.gradient_learner import GradientMetaLearner


class Maml_TRPO(GradientMetaLearner):
    def __init__(self, policy: Policy, spec: RunningSpec):
        """

        :param policy:
        :param spec:
        """
        super(Maml_TRPO, self).__init__(policy, spec)
        assert self.policy is not None
        assert self.spec is not None

        self.first_order = self.spec.get('first_order', 'meta_task')

    def adapt(self, train_futures, first_order=None):
        """

        :param train_futures:
        :param first_order:
        :return:
        """
        if first_order is None:
            first_order = self.first_order

        # Loop over the number of steps of adaptation
        params = None
        for e in train_futures:
            inner_loss = reinforce_loss(self.policy, e, W=params)
            params = self.policy.update_params(inner_loss,
                                               params=params,
                                               step_size=self.fast_lr,
                                               first_order=first_order)
        return params

    def surrogate_loss(self, train_futures, valid_futures, old_policy=None):
        """
        :param old_policy:
        :param train_futures:
        :param valid_futures:
        :return:
        """
        first_order = (old_policy is not None) or self.first_order
        params = self.adapt(train_futures, first_order=first_order)

        with torch.set_grad_enabled(old_policy is None):
            valid_episodes = valid_futures
            pi = self.policy(valid_episodes.observations.float(), W=params)

            if old_policy is None:
                old_pi = detach_dist_from_policy(pi)

            log_ratio = (pi.log_prob(valid_episodes.actions) - old_pi.log_prob(valid_episodes.actions))
            ratio = torch.exp(log_ratio)

            losses = -weighted_mean(ratio * valid_episodes.advantages, lengths=valid_episodes.lengths)
            kls = weighted_mean(kl_divergence(pi, old_pi), lengths=valid_episodes.lengths)

        return losses.mean(), kls.mean(), old_pi

    def step(self, train_futures, valid_futures):

        # max_kl = 1e-3,
        # cg_iters = 10,
        # cg_damping = 1e-2,
        # ls_max_steps = 10,
        # ls_backtrack_ratio = 0.5
        #

        print(f"{self.max_kl} {self.cg_iters} {self.cg_damping} {self.ls_max_steps} {self.ls_backtrack_ratio}")

        num_tasks = len(train_futures)
        logs = {}

        print("num_tasks", num_tasks)

        # Compute the surrogate loss
        print(len(train_futures))
        print(len(valid_futures))
        # for (train, valid) in zip(train_futures, valid_futures)
        # old_losses, old_kls, old_pis = [self.surrogate_loss(train, valid, old_policy=None)
        #                                 for (train, valid) in
        #                                 zip(train_futures, valid_futures)]

        old_losses, old_kls, old_pis = [self.surrogate_loss(t, v, old_policy=None)
                                        for (t, v) in zip(train_futures, valid_futures)]
        #     old_losses, old_kls, old_pis = self.surrogate_loss(train_futures, v, old_policy=None)
        #     print(old_losses)
        #     print(old_kls)
        #     print(old_pis)
        #
        # raise

        logs['loss_before'] = to_numpy(old_losses)
        logs['kl_before'] = to_numpy(old_kls)

        print(f"Old loss {old_losses}")
        print(f"Old num_tasks {num_tasks}")

        old_loss = sum(old_losses) / num_tasks
        grads = torch.autograd.grad(old_loss, self.policy.parameters(), retain_graph=True)
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        old_kl = sum(old_kls) / num_tasks
        hessian_vector_product = self.hessian_vector_product(old_kl)
        stepdir = conjugate_gradient(hessian_vector_product,
                                     grads,
                                     cg_iters=self.cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir,
                              hessian_vector_product(stepdir, retain_graph=False))
        lagrange_multiplier = torch.sqrt(shs / self.max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(self.ls_max_steps):
            vec2parameters(old_params - step_size * step, self.policy.parameters())
            losses, kls, _ = [
                self.surrogate_loss(train, valid, old_pi=old_pi)
                for (train, valid, old_pi)
                in zip(zip(*train_futures), valid_futures, old_pis)]

            improve = (sum(losses) / num_tasks) - old_loss
            kl = sum(kls) / num_tasks
            if (improve.item() < 0.0) and (kl.item() < self.max_kl):
                logs['loss_after'] = to_numpy(losses)
                logs['kl_after'] = to_numpy(kls)
                break
            step_size *= self.ls_backtrack_ratio
        else:
            vec2parameters(old_params, self.policy.parameters())

        return logs
