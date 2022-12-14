# Tuple used to communicate between RPC speakers, for now RPC. I have to detach tensor
# So on same node I still need detach.  Hopefully torch will add CUDA memory sharing option
# for a same node.
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
                           "reward_dtype",
                           "action_dtype",
                           "observations_dtype"))

TrajectoryData = namedtuple("TrajectoryData", ("train", "validate"))
