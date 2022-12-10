from typing import Any

import torch
from torch import dtype

#
# for reward, batch_id in zip(rewards, batch_ids):
#     if batch_id is None:
#         continue
#     rewards_list[batch_id] = torch.as_tensor(reward)
#     batch_idx.append(batch_id)
#
# rewards = torch.stack(rewards_list, dim=0)
# print(rewards.shape)
# _rewards = torch.zeros((1, 5))
#
# #
# _rewards = torch.stack((rewards, _rewards), dim=0, out=_rewards[:_rewards.shape[1], _rewards.shape[5]])
# # print(self._rewards.shape)
#
# # Have a list of tensors (which can be of different lengths)
# data = [torch.tensor([-0.1873, -0.6180, -0.3918, -0.5849, -0.3607]),
#         torch.tensor([-0.6873, -0.3918, -0.5849, -0.9768, -0.7590, -0.6707]),
#         torch.tensor([-0.6686, -0.7022, -0.7436, -0.8231, -0.6348, -0.4040, -0.6074, -0.6921])]
#
# # # Determine maximum length
# # max_len = max([x.squeeze().numel() for x in data])
# # # pad all tensors to have same length
# # data = [torch.nn.functional.pad(x, pad=(0, max_len - x.numel()), mode='constant', value=0) for x in data]
#
# # stack them
# data = torch.stack(data)
#
# print(data)
# print(data.shape)


from typing import List, Any, Optional

import numpy as np
import torch


class BaseBatchEpisodes(object):
    def __init__(self, batch_size, device: Optional[torch.device] = 'cpu'):
        """
        :param batch_size:
        :param device:
        """
        self.lengths1 = None
        self._lengths2 = None
        self.batch_size = batch_size
        self.device = device

        self._observation_shape = None
        self._action_shape = None

        self._observations = None
        self._actions = None
        self._rewards = None
        self._lengths = None

        self._actions_list = [[] for _ in range(batch_size)]
        self._observations_list = [[] for _ in range(batch_size)]
        self._rewards_list = [[] for _ in range(batch_size)]

    def append(self, observations: np.ndarray, actions, rewards: Any, batch_ids: List[int]):
        """
        :param observations:
        :param actions:
        :param rewards:
        :param batch_ids:
        :return:
        """
        if observations is None:
            raise ValueError("Observations is none")

        if actions is None:
            raise ValueError("action is none")

        for observation, action, reward, batch_id in zip(observations, actions, rewards, batch_ids):
            if batch_id is None:
                continue

            self._observations_list[batch_id].append(torch.as_tensor(observation))
            self._actions_list[batch_id].append(torch.as_tensor(action))
            self._rewards_list[batch_id].append(torch.as_tensor(reward))

    @property
    def rewards(self) -> torch.Tensor:
        """
        :return:
        """
        if self._rewards is not None:
            return self._rewards

        rewards = torch.zeros((len(self), self.batch_size), device=self.device, dtype=torch.float32)
        for i in range(self.batch_size):
            length = self.lengths[i]
            if length > 0:
                torch.stack(self._rewards_list[i], dim=0, out=rewards[:length, i])
        self._rewards = torch.as_tensor(rewards, device=self.device)
        del self._rewards_list
        return self._rewards

    @property
    def lengths(self):
        """
        :return:
        """
        if self._lengths2 is not None:
            return self._lengths2

        self._lengths2 = [len(r) for r in self._rewards_list]
        self.lengths1 = torch.zeros([len(self._rewards_list), 1],
                                    dtype=torch.int32, device=self.device,
                                    requires_grad=False)
        for i, rewards_t in enumerate(self._rewards_list):
            self.lengths1[i] = len(rewards_t)
        self.lengths1 = self.lengths1.T

        print(self.lengths1)
        return self._lengths2

    def __len__(self):
        """
        Return largest episode len
        :return:
        """
        print(torch.max(self.lengths1).item())
        return max(self.lengths)


b = BaseBatchEpisodes(batch_size=5)
rewards = np.asarray([10])
obs = np.asarray([10])
act = np.asarray([1])
batch_ids = [0]
rewards_list = [Any] * 5

# for i in range(len(obs)):
b.append(obs, act, rewards, batch_ids)

rewards = np.asarray([1])
obs = np.asarray([1])
act = np.asarray([1])
batch_ids = [0]
b.append(obs, act, rewards, batch_ids)

rewards = np.asarray([2])
obs = np.asarray([1])
act = np.asarray([1])
batch_ids = [0]
b.append(obs, act, rewards, batch_ids)

rewards = np.asarray([1])
obs = np.asarray([1])
act = np.asarray([1])
batch_ids = [1]
b.append(obs, act, rewards, batch_ids)

rewards = np.asarray([1])
obs = np.asarray([1])
act = np.asarray([1])
batch_ids = [2]
b.append(obs, act, rewards, batch_ids)

rewards = np.asarray([1])
obs = np.asarray([1])
act = np.asarray([1])
batch_ids = [3]
b.append(obs, act, rewards, batch_ids)

# rewards = np.asarray([1])
# obs = np.asarray([1])
# act = np.asarray([1])
# batch_ids = [4]
# b.append(obs, act, rewards, batch_ids)

print(b.lengths)
print(b.lengths1)
print(len(b))
print(b.rewards)
print(b.rewards.shape)
