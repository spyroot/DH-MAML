"""
This base class so we cna different version.
Mus
"""
from typing import List, Any, Optional

import numpy as np
import torch


class BaseBatchEpisodes(object):
    def __init__(self, batch_size, device: Optional[torch.device] = 'cpu'):
        """
        This base class for batched version.
        :param batch_size:
        :param device:
        """
        self.batch_size = batch_size
        self.device = device

        self._observation_shape = None
        self._action_shape = None

        self._observations = None
        self._actions = None
        self._rewards = None
        self._returns = None
        self._mask = None
        self._advantages = None
        self._lengths = None

        self._observations_list = [[] for _ in range(batch_size)]
        self._actions_list = [[] for _ in range(batch_size)]
        self._rewards_list = [[] for _ in range(batch_size)]

    def __str__(self):
        """
        :return:
        """
        return f"{self._observations}" + f"{self._actions}"

    def batch_info(self):
        print(f"batch size={self.batch_size}")
        print(f"num actions {len(self._actions_list)}")
        print(f"num rewards {len(self._rewards_list)}")
        print(f"num observation {len(self._observations_list)}")
        print(f"observation shape {self.observation_shape}")
        print(f"action shape {self.action_shape}")

    @property
    def observation_shape(self):
        if self._observation_shape is None:
            self._observation_shape = self.observations.shape[2:]
        return self._observation_shape

    @property
    def action_shape(self):
        if self._action_shape is None:
            self._action_shape = self.actions.shape[2:]
        return self._action_shape

    @property
    def observations(self) -> torch.Tensor:
        """
        :return:
        """
        if self._observations is not None:
            return self._observations

        observation_shape = self._observations_list[0][0].shape
        obs_dtype = self._observations_list[0][0].dtype
        obs = np.zeros((len(self), self.batch_size) + observation_shape, dtype=obs_dtype)
        for i in range(self.batch_size):
            length = self.lengths[i]
            # print(f"{i}, {obs[:length, i]}")
            np.stack(self._observations_list[i], axis=0, out=obs[:length, i])

        self._observations = torch.as_tensor(obs, device=self.device)
        del self._observations_list

        self._observations = self.observations.to(self.device)
        return self._observations

    @property
    def actions(self) -> torch.Tensor:
        """

        :return:
        """
        if self._actions is not None:
            return self._actions

        action_shape = self._actions_list[0][0].shape
        action_dtype = self._actions_list[0][0].dtype
        assert len(self) > 0
        _actions = torch.zeros(((len(self), self.batch_size) + action_shape), dtype=action_dtype,
                               requires_grad=False, device=self.device)

        # print(len(self._actions_list))
        # print(self._actions_list[0].shape)

        for i in range(self.batch_size):
            torch.stack(self._actions_list[i], dim=0, out=_actions[:self._lengths[i], i])

        # self._actions = torch.as_tensor(actions, device=self.device)
        self._actions = _actions.to(self.device)
        del self._actions_list
        return self._actions

    def get_action(self):
        return self._actions

    @property
    def rewards(self) -> torch.Tensor:
        """
        :return:
        """
        if self._rewards is not None:
            return self._rewards

        rewards = torch.zeros((len(self), self.batch_size),
                              device=self.device, dtype=torch.float32)

        for i in range(self.batch_size):
            length = self.lengths[i]
            torch.stack(self._rewards_list[i], dim=0, out=rewards[:length, i])

        self._rewards = torch.as_tensor(rewards, device=self.device)
        del self._rewards_list
        return self._rewards

    @property
    def mask(self) -> torch.Tensor:
        """
        :return:
        """
        if self._mask is not None:
            return self._mask

        self._mask = torch.zeros((len(self), self.batch_size),
                                 dtype=torch.float32,
                                 device=self.device)

        for i in range(self.batch_size):
            length = self.lengths[i]
            self._mask[:length, i].fill_(1.0)

        self._mask = self._mask.to(self.device)
        return self._mask

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

            self._observations_list[batch_id].append(observation.astype(observation[0].dtype))
            action = torch.as_tensor(action.astype(actions.dtype)).to(self.device)

            self._actions_list[batch_id].append(action.to(self.device))

            reward = torch.as_tensor(np.array(reward).astype('float')).to(self.device)
            self._rewards_list[batch_id].append(reward)

        self.update_actions()
        self._lengths = torch.zeros([len(self._rewards_list), 1], dtype=torch.int32,
                                    device=self.device, requires_grad=False)

        for i, rewards in enumerate(self._rewards_list):
            self._lengths[i] = len(rewards)

    @property
    def lengths(self):
        """
        :return:
        """
        if self._lengths is not None:
            return self._lengths

        self._lengths = torch.zeros([len(self._rewards_list), 1], dtype=torch.int32,
                                    device=self.device, requires_grad=False)
        for i, rewards in enumerate(self._rewards_list):
            self._lengths[i] = len(rewards)
        return self._lengths

    def __len__(self):
        """Return largest episode length. maybe truncated
        :return:
        """
        if self._lengths is None:
            return torch.max(self.lengths)

        return torch.max(self._lengths)
