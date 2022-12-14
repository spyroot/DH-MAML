# Tuple used to communicate between for RPC.
from collections import namedtuple

NamedEpisode = namedtuple("NamedEpisode",
                          ("observations",
                           "actions",
                           "rewards",
                           "lengths",
                           "advantages",
                           "returns",
                           "mask",
                           "action_shape",
                           "observation_shape",
                           "max_len",
                           "batch_size",
                           reward_dtype,
                           action_dtype,
                           observations_dtype))

TrajectoryData = namedtuple("TrajectoryData", ("train", "validate"))
