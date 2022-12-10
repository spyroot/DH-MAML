from typing import Union, Iterator, Callable, Optional

import gym
from gym import Env
from gym.envs.registration import EnvSpec
from torch.utils.tensorboard import SummaryWriter
from meta_critics.wrappers.record_tf_episode_statistics import RecordTensorboardEpisodeStatistics


def make_env(gym_id: Union[str, EnvSpec], seed,
             idx: int, capture_video: bool, run_name: str,
             continuous: Optional[bool] = False, enable_wind: Optional[bool] = False,
             episode_trigger: Optional[Callable[[int], bool]] = None,
             episode_log_trigger: Optional[Callable] = None):
    """
    :param episode_log_trigger:
    :param gym_id:
    :param seed:
    :param idx:
    :param capture_video:
    :param run_name:
    :param continuous:
    :param enable_wind:
    :param episode_trigger:
    :return:
    """

    def creator() -> Env:
        """

        :return:
        """
        env = gym.make(gym_id, render_mode="rgb_array", continuous=continuous, enable_wind=enable_wind)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            if episode_trigger is not None:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=episode_trigger)

        writer = SummaryWriter(f"runs/{run_name}")
        env = RecordTensorboardEpisodeStatistics(env, writer, episode_log_trigger)
        env.action_space.seed(seed)
        return env

    return creator


def create_env(gym_id: Union[str, EnvSpec],
               seed: Optional[int] = 0,
               num_envs: Optional[int] = 1,
               capture_video: Optional[bool] = False,
               run_name: Optional[str] = "test",
               episode_rec_trigger: Optional[Callable[[int], bool]] = None,
               episode_log_trigger: Optional[Callable] = None,
               ) -> Iterator[Callable[[], Env]]:
    """

    :param gym_id:
    :param seed:
    :param num_envs:
    :param capture_video:
    :param run_name:
    :param episode_log_trigger:
    :param episode_rec_trigger:
    :return:
    """
    for i in range(num_envs):
        yield make_env(gym_id, seed + 1, i, capture_video, run_name,
                       continuous=False, enable_wind=False,
                       episode_trigger=episode_rec_trigger,
                       episode_log_trigger=episode_log_trigger)
