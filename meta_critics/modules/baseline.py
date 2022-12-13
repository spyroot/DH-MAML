"""
Linear baseline based on handcrafted features, as described in [1]
(Supplementary Material 2).

[1] Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel,
    "Benchmarking Deep Reinforcement Learning for Continuous Control", 2016
    (https://arxiv.org/abs/1604.06778)
"""
from typing import Optional

import torch
import torch.nn as nn
from meta_critics.trajectory.base_trajectory import BaseTrajectory
from functools import reduce
from operator import mul


class LinearFeatureBaseline(nn.Module):
    def __init__(self, env, device: torch.device, reg_coefficient: Optional[float] = 1e-5):
        """
        :param reg_coefficient:
        """
        super(LinearFeatureBaseline, self).__init__()
        self._env = env
        self._device = device
        self._reg_coefficient = reg_coefficient
        self._input_size = reduce(mul, env.observation_space.shape, 1)
        self.weight = nn.Parameter(torch.Tensor(self.feature_size, ),
                                   requires_grad=False).to(self._device)
        self.weight.data.zero_()
        self._eye = torch.eye(self.feature_size, dtype=torch.float32,
                              device=self.weight.device, requires_grad=False).to(self._device)

    @property
    def feature_size(self) -> int:
        return 2 * self._input_size + 4

    def _feature(self, episodes: BaseTrajectory) -> torch.Tensor:
        """
        :param episodes:
        :return:
        """
        ones = episodes.mask.unsqueeze(2).to(self._device)
        observations = episodes.observations
        lens = torch.arange(torch.max(episodes.lengths).item(), device=self._device)
        time_step = lens.view(-1, 1, 1) * ones / 100.0

        t = torch.cat([
            observations,
            observations ** 2,
            time_step,
            time_step ** 2,
            time_step ** 3,
            ones
        ], dim=2)

        t.to(self._device)
        return t

    def fit(self, episodes: BaseTrajectory):
        """
        """
        # sequence_length * batch_size x feature_size
        feature_matrix = self._feature(episodes).view(-1, self.feature_size)  # seq_len * batch_size x feature_size
        returns = episodes.returns.view(-1, 1)  # sequence_length * batch_size x 1

        if torch.all(returns == 0):
            return

        flat_mask = episodes.mask.flatten()
        flat_mask_nnz = torch.nonzero(flat_mask)

        # print(episodes.mask)
        # print("FLAT MASK", flat_mask)

        feature_matrix = feature_matrix[flat_mask_nnz].view(-1, self.feature_size)
        returns = returns[flat_mask_nnz].view(-1, 1)
        reg_coeff = self._reg_coefficient

        XT_y = torch.matmul(feature_matrix.t(), returns)
        XT_X = torch.matmul(feature_matrix.t(), feature_matrix)
        assert XT_X.shape == self._eye.shape
        assert XT_y.shape[0] == XT_X.shape[0] == self._eye.shape[0]

        for _ in range(5):
            try:
                coeffs = torch.linalg.lstsq(XT_y, XT_X + reg_coeff * self._eye).solution
                if torch.isnan(coeffs).any() or torch.isinf(coeffs).any():
                    print("nan")
                    raise RuntimeError
                break
            except RuntimeError as re:
                print("runtime", re)
                reg_coeff *= 10
        else:
            print(f'Unable to solve the normal equations {reg_coeff}.')
            print("returns.dtype", returns.dtype)
            print("feature_matrix.dtype", feature_matrix.dtype)
            print("returns ", returns)
            print("return", feature_matrix)
            raise RuntimeError("Unable to solve the normal equations")

        self.weight.copy_(coeffs.flatten())

    def forward(self, episodes: BaseTrajectory):
        """
        :param episodes:
        :return:
        """

        features = self._feature(episodes)
        values = torch.mv(features.view(-1, self.feature_size).float(), self.weight)
        return values.view(features.shape[:2])
