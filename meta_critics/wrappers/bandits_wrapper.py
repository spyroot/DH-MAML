from gym.envs.registration import load

from meta_critics.wrappers.wrapper_normalize_env import NormalizedActionWrapper


def bandits_wrapper(entry_point, **kwargs):
    """
    :param entry_point:
    :param kwargs:
    :return:
    """
    normalization_scale = kwargs.pop('normalization_scale', 1.)
    kwargs['render_mode'] = None
    env_cls = load(entry_point)
    env = env_cls(**kwargs)
    env = NormalizedActionWrapper(env, scale=normalization_scale)
    return env
