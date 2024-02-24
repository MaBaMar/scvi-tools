from abc import abstractmethod

import torch
from torch import nn


class BasePrior(nn.Module):

    @property
    @abstractmethod
    def distribution(self):
        ...

    @abstractmethod
    def sample(self, n_samples: int) -> torch.Tensor:
        ...

    @abstractmethod
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        ...
    
    def description(self):
        return "Base Prior parent class of all Prior classes."
