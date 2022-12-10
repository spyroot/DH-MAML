import torch
from torch.distributions import Categorical, Independent, Normal, LogNormal


def detach_module(module, keep_requires_grad=False):
    """

    :param module:
    :param keep_requires_grad:
    :return:
    """
    if not isinstance(module, torch.nn.Module):
        return
    for param_key in module._parameters:
        if module._parameters[param_key] is not None:
            requires_grad = module._parameters[param_key].requires_grad
            detached = module._parameters[param_key].detach_()
            if keep_requires_grad and requires_grad:
                module._parameters[param_key].requires_grad_()

    for buffer_key in module._buffers:
        if module._buffers[buffer_key] is not None and \
                module._buffers[buffer_key].requires_grad:
            module._buffers[buffer_key] = module._buffers[buffer_key].detach_()
            if keep_requires_grad:  # requires_grad checked above
                module._buffers[buffer_key].requires_grad_()

    for module_key in module._modules:
        detach_module(module._modules[module_key], keep_requires_grad=keep_requires_grad)


def detach_dist_from_policy(agent_policy, device):
    """
    :param device:
    :param agent_policy:
    :return:
    """
    if isinstance(agent_policy, Independent):
        distribution = Independent(detach_dist_from_policy(agent_policy.base_dist, device),
                                   agent_policy.reinterpreted_batch_ndims)
    elif isinstance(agent_policy, Categorical):
        distribution = Categorical(logits=agent_policy.logits.detach().to(device=device))
    elif isinstance(agent_policy, LogNormal):
        distribution = LogNormal(loc=agent_policy.loc.detach().to(device),
                                 scale=agent_policy.scale.detach().to(device))
    elif isinstance(agent_policy, Normal):
        distribution = Normal(loc=agent_policy.loc.detach().to(device),
                              scale=agent_policy.scale.detach().to(device))
    else:
        raise NotImplementedError("Wrong type policy")
    return distribution


def detach_dist(dist):
    for param_key in dist.__dict__:
        item = dist.__dict__[param_key]
        if isinstance(item, torch.Tensor):
            if item.requires_grad:
                dist.__dict__[param_key] = dist.__dict__[param_key].detach()
        elif isinstance(item, torch.nn.Module):
            dist.__dict__[param_key] = detach_module(dist.__dict__[param_key])
        elif isinstance(item, torch.distributions.Distribution):
            dist.__dict__[param_key] = detach_dist(dist.__dict__[param_key])
    return dist
