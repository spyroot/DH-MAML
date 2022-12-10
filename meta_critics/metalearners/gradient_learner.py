from typing import Optional, Callable

import torch
from torch.nn.utils import parameters_to_vector

from meta_critics.policies.policy import Policy
from meta_critics.running_spec import RunningSpec


class GradientMetaLearner(object):
    def __init__(self, policy: Policy, spec: RunningSpec, debug=False):
        """

        :param policy: Agent Policy Meta Learner will use
        :param spec:  Specs for meta learner. Different Meta Learner might have own model specific.
        """

        self.policy = policy
        self.spec = spec

        assert self.policy is not None
        assert self.spec is not None

        # shared parameters
        self.max_kl = self.spec.get('max_kl', 'model')
        self.cg_iters = self.spec.get('cg_iters', 'model')
        self.cg_damping = self.spec.get('cg_damping', 'model')
        self.ls_max_steps = self.spec.get('ls_max_steps', 'model')
        self.ls_backtrack_ratio = self.spec.get('ls_backtrack_ratio', 'model')
        self.fast_lr = self.spec.get('fast_lr', 'meta_task')

        self.device = torch.device(self.spec.get('device'))
        self.policy.to(self.device)
        self.debug = debug

        if debug:
            print(f"{self.max_kl} {self.cg_iters} {self.ls_max_steps} {self.ls_backtrack_ratio} {self.device}")

    def adapt(self, episodes, first_order=Optional[bool]):
        raise NotImplementedError()

    def step(self, train_episodes, valid_episodes):
        raise NotImplementedError()

    def hessian_vector_product(self, kl: torch.Tensor) -> Callable:
        """
        :param kl:
        :return:
        """
        flat_grad_kl = parameters_to_vector(torch.autograd.grad(kl, list(self.policy.parameters()), create_graph=True))

        def v_dot_grad(vector: torch.Tensor, retain_graph=True) -> torch.Tensor:
            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = parameters_to_vector(torch.autograd.grad(grad_kl_v, list(self.policy.parameters()),
                                                              retain_graph=retain_graph))
            return grad2s + self.cg_damping * vector

        return v_dot_grad
