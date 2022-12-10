"""
This base class so we cna different version.
Mus
"""
from abc import abstractmethod
import torch


class BaseTrajectory:
    def __init__(self):
        self._returns = None
        self._lengths = None
        self._mask = None
        self._observations = None

    @property
    @abstractmethod
    def returns(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def lengths(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def mask(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def observations(self) -> torch.Tensor:
        pass
