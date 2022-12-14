from copy import deepcopy
from typing import Optional, Any, List

import numpy as np
import torch
import torch.nn.functional as F

from meta_critics.base_trainer.torch_tools.torch_utils import weighted_normalize
from meta_critics.base_trainer.torch_tools.tensor_tools import numpy_to_torch_dtype_dict, numpy_to_torch_remaping
from meta_critics.ioutil.term_util import print_red
from meta_critics.trajectory.base_trajectory import BaseTrajectory
from meta_critics.named_episode import NamedEpisode


class AdvantageBatchEpisodes(BaseTrajectory):
    def __init__(self,
                 batch_size: Optional[int],
                 gamma: Optional[float] = 0.95,
                 device: Optional[torch.device] = 'cuda',
                 advantages=None,
                 returns=None,
                 actions=None,
                 observations=None,
                 action_shape=None,
                 observation_shape=None,
                 lengths=None,
                 rewards=None,
                 mask=None,
                 debug=False,
                 remap_dtype=False,
                 reward_dtype: Optional[Any] = torch.float32,
                 action_dtype: Optional[Any] = torch.float32,
                 observations_dtype: Optional[Any] = torch.float32,
                 ):
        """

        :param batch_size:
        :param gamma:
        :param device:
        """
        super(AdvantageBatchEpisodes, self).__init__()
        self.debug = debug

        if self.debug and observations is not None:
            print("Creating from args")

        # self.lock = lock
        self._action_dtype = action_dtype
        self._reward_dtype = reward_dtype
        self._observation_dtype = observations_dtype

        self.max_len = None
        self.gamma = gamma
        self.is_copy = False
        self.is_full = False
        self.is_remap_floats = remap_dtype

        self.batch_size = batch_size
        self.device = device

        self._advantages = advantages
        self._observation_shape = None
        self._action_shape = None

        if observations is not None:
            self.is_copy = True

        self._observations = observations
        self._advantages = advantages
        self._actions = actions
        self._rewards = rewards
        self._returns = returns
        self._lengths = lengths
        self._mask = mask

        if self._actions is not None:
            self._action_shape = self._actions.shape[2:]

        if self._observations is not None:
            self._observation_shape = self._observations.shape[2:]

        if self._actions is None:
            self._actions_list = [[] for _ in range(batch_size)]

        if self._observations is None:
            self._observations_list = [[] for _ in range(batch_size)]

        if self._rewards is None:
            self._rewards_list = [[] for _ in range(batch_size)]

    def to_gpu(self):
        # with self.lock:
        # print("to gpu is a copy ", self.is_copy)
        self._observations = self._observations.to(self.device)
        if self._actions is not None:
            self._actions = self._actions.to(self.device)
        self._rewards = self._rewards.to(self.device)
        self._lengths = self._lengths.to(self.device)
        self._advantages = self._advantages.to(self.device)
        self._returns = self._returns.to(self.device)
        # self._mask = self._mask.to(self.device)

    def clone_as_tuple(self):
        """
        :return:
        """
        # with self.lock:
        observations = self._observations.clone().cpu()
        actions = self.actions.clone().detach().cpu()
        rewards = self._rewards.clone().detach().cpu()
        lengths = self._lengths.clone().detach().cpu()
        advantages = self._advantages.clone().cpu()
        returns = self._returns.clone().cpu()
        mask = self._mask.clone().cpu()
        action_shape = deepcopy(self._action_shape)
        observation_shape = deepcopy(self._observation_shape)
        max_len = deepcopy(self.max_len)
        batch_size = deepcopy(self.batch_size)
        reward_dtype = deepcopy(self._reward_dtype)
        action_dtype = deepcopy(self._action_dtype)
        observation_dtype = deepcopy(self._observation_dtype)

        return NamedEpisode(observations, actions, rewards, lengths, advantages,
                            returns, mask, action_shape, observation_shape,
                            max_len, batch_size, reward_dtype, action_dtype, observation_dtype)

    def clone(self, x):
        """
        :return:
        """
        # with self.lock:
        x._observations = self._observations.clone().cpu()
        x._actions = self._actions.clone().detach().cpu()
        x._rewards = self._rewards.clone().detach().cpu()
        x._lengths = self._lengths.clone().detach().cpu()
        x._advantages = self._advantages.clone().cpu()
        x._returns = self._returns.clone().cpu()
        x._mask = self._mask.clone().cpu()
        x._action_shape = deepcopy(self._action_shape)
        x._observation_shape = deepcopy(self._observation_shape)
        x.max_len = self.max_len
        x.batch_size = self.batch_size
        x.gamma = self.gamma
        x._reward_dtype = deepcopy(self._reward_dtype)
        x._action_dtype = deepcopy(self._action_dtype)
        x._observation_dtype = deepcopy(self._observation_dtype)
        return x

    # def bootstrap(self):
    # with torch.no_grad():
    #     next_value = agent.get_value(next_obs).reshape(1, -1)
    #     advantages = torch.zeros_like(rewards).to(device)
    #     lastgaelam = 0
    #     for t in reversed(range(args.num_steps)):
    #         if t == args.num_steps - 1:
    #             nextnonterminal = 1.0 - next_done
    #             nextvalues = next_value
    #         else:
    #             nextnonterminal = 1.0 - dones[t + 1]
    #             nextvalues = values[t + 1]
    #         delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
    #         advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
    #     returns = advantages + values
    @property
    def advantages(self):
        """
        :return:
        """
        if self._advantages is None:
            raise ValueError('The advantages have not been computed. .')
        return self._advantages

    @property
    def returns(self):
        """
        :return:
        """
        if self._returns is not None:
            return self._returns

        self._returns = torch.zeros_like(self.rewards, device=self.device)
        return_ = torch.zeros((self.batch_size,), dtype=self._reward_dtype, device=self.device)
        for i in range(len(self) - 1, -1, -1):
            return_ = self.gamma * return_ + self.rewards[i] * self.mask[i]
            self._returns[i] = return_

        return self._returns

    def recompute_advantages(self, baseline, gae_lambda=1.0, normalize=True, debug=False):
        """
        :param debug:
        :param baseline:
        :param gae_lambda:
        :param normalize:
        :return:
        """
        # Compute the values based on the baseline
        # assert self._advantages is None
        values = baseline(self).detach().clone()
        values = F.pad(values * self.mask, (0, 0, 0, 1))
        if debug:
            print_red(f"reward tensor {self.rewards}")

        deltas = self.rewards + self.gamma * values[1:] - values[:-1]
        self._advantages = torch.zeros_like(self.rewards, device=self.device)
        gae = torch.zeros((self.batch_size,), device=self.device, dtype=self._reward_dtype)
        if debug:
            print_red(f"gae {gae}")

        for i in range(torch.max(self._lengths) - 1, -1, -1):
            gae = gae * self.gamma * gae_lambda + deltas[i]
            self._advantages[i] = gae

        if normalize:
            self._advantages = weighted_normalize(self._advantages, lengths=self.lengths)

        if debug:
            print_red(f"final advantage vector {self._advantages}")

        return self._advantages

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
            np.stack(self._observations_list[i], axis=0, out=obs[:self._lengths[i], i])
        self._observations = torch.as_tensor(obs, device=self.device)
        # self._observations_list
        return self._observations

    @property
    def actions(self) -> torch.Tensor:
        """
        :return:
        """
        if self._actions is not None:
            return self._actions

        action_shape = self._actions_list[0][0].shape
        assert len(self) > 0
        _actions = torch.zeros(((len(self), self.batch_size) + action_shape), dtype=self._action_dtype,
                               requires_grad=False, device=self.device)

        for i in range(self.batch_size):
            torch.stack(self._actions_list[i], dim=0, out=_actions[:self._lengths[i], i])

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

        rewards = torch.zeros((len(self), self.batch_size), device=self.device, 
                              requires_grad=False, dtype=self._reward_dtype)
        for i in range(self.batch_size):
            torch.stack(self._rewards_list[i], dim=0, out=rewards[:self.lengths[i], i])

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
                                 dtype=torch.float32, requires_grad=False, device=self.device)
        for i in range(self.batch_size):
            self._mask[:self._lengths[i], i].fill_(1.0)

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
            raise ValueError("Observations is none.")
        if actions is None:
            raise ValueError("action is none")

        if self.is_remap_floats:
            type_remap = numpy_to_torch_remaping
        else:
            type_remap = numpy_to_torch_dtype_dict

        for observation, action, reward, batch_id in zip(observations, actions, rewards, batch_ids):
            if batch_id is None:
                continue

            if self._observation_dtype is None:
                dt: np.dtype = observation.dtype
                self._observation_dtype = type_remap[dt.type]

            if self._action_dtype is None:
                if np.isscalar(action):
                    dt: np.dtype = np.dtype(action)
                else:
                    dt: np.dtype = np.dtype(action[-1])
                self._action_dtype = type_remap[dt.type]

            if self._reward_dtype is None:
                dt: np.dtype = np.dtype(reward)
                self._reward_dtype = type_remap[dt.type]

            self._observations_list[batch_id].append(observation.astype(observation[0].dtype))
            self._actions_list[batch_id].append(torch.as_tensor(action, device=self.device, dtype=self._action_dtype))
            self._rewards_list[batch_id].append(torch.as_tensor(reward, device=self.device, dtype=self._reward_dtype))

        self.is_full = True

    @property
    def lengths(self):
        """
        The size of trajectory
        :return:
        """
        if self._lengths is not None:
            return self._lengths

        self._lengths = torch.zeros([len(self._rewards_list), 1],
                                    dtype=torch.int32, device=self.device,
                                    requires_grad=False)
        for i, rewards_t in enumerate(self._rewards_list):
            self._lengths[i] = len(rewards_t)

        # self._lengths = self._lengths.T
        return self._lengths

    def __len__(self):
        """
        Return largest episode len
        :return:
        """
        return torch.max(self.lengths).item()

    def require_grad(self):
        """
        Mainly for debug in  case of multiprocessing
        :return:
        """
        if self._observations.requires_grad:
            print(f"observations required grad {self._observations.requires_grad}")

        if self._actions is not None and self._actions.requires_grad:
            print(f"action required grad {self._actions.requires_grad}")

        if self._rewards.requires_grad:
            print(f"action required grad {self._rewards.requires_grad}")

        if self._returns.requires_grad:
            print(f"action required grad {self._returns.requires_grad}")

        if self._advantages.requires_grad:
            print(f"action required grad {self._advantages.requires_grad}")

        if self.lengths.requires_grad:
            print(f"action required grad {self._lengths.requires_grad}")

        if self._mask.requires_grad:
            print(f"action required grad {self._mask.requires_grad}")
