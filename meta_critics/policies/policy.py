from abc import abstractmethod
from typing import Optional
import torch
import torch.nn as nn

from collections import OrderedDict


class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        """
        :param input_size:
        :param output_size:
        """
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.meta_parameters = None
        # self.sigma = nn.Parameter(torch.Tensor(1))
        # self.register_parameter("sigma", self.sigma)

    def register_param_names(self):
        self.meta_parameters = self.named_parameters

    def update_grads(self, param_dict):
        """
        :param param_dict:
        :return:
        """
        for new_param_key, new_param_val, in param_dict.items():
            for self_param_key, self_param_val, in self.meta_parameters():
                if new_param_key == self_param_key:
                    if new_param_val.grad is not None:
                        self_param_val.grad = new_param_val.grad.clone()

    def update_parameters(self, param_dict):
        """ This update without grad.
        :param param_dict:
        :return:
        """
        # we just copy to be on save side.
        for name, params in self.meta_parameters():
            params.data.copy_(param_dict[name])

    def update_params(self, loss, params=None, step_size: Optional[float] = 0.5, first_order=False):
        """This will one grad with step size
        :param loss:
        :param params:
        :param step_size:
        :param first_order:
        :return:
        """
        if params is None:
            params = OrderedDict(self.meta_parameters())

        t = list(params.values())
        grads = torch.autograd.grad(loss, t, create_graph=not first_order)

        updated_params = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size * grad
            # print("updates", updated_params[name].gradient)

        return updated_params

    def set_parameters(self, parameters):
        print("received parameters")
        # self.parameters = parameters
