from gym.envs.registration import load
from gym.wrappers import TimeLimit


def nav_wrapper(entry_point, **kwargs):
    """Wrapper called to create 2d nav env
    :param entry_point:
    :param kwargs:
    :return:
    """
    # normalization_scale = kwargs.pop('normalization_scale', 1.)
    max_episode_steps = kwargs.pop('max_episode_steps', 100)

    # Load the environment from its entry point
    env_cls = load(entry_point)
    env = env_cls(**kwargs)
    # Normalization wrapper
    # env = NormalizedActionWrapper(env, scale=normalization_scale)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env
