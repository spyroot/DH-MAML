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
                           "batch_size"))

TrajectoryData = namedtuple("TrajectoryData", ("train", "validate"))
