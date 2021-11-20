"""Utils."""

from typing import List
from torch import Tensor

import torch
from torch import nn


def initialize_weights(layer: nn.Module, mean: float = 0.0, std: float = 0.02):
    """Initialize module with normal distribution.

    Parameters
    ----------
    layer: nn.Module
        Layer.
    mean: float, (default=0.0)
        Mean value.
    std: float, (default=0.02)
        Standard deviation value.

    """
    if isinstance(layer, (nn.Conv3d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(layer.weight, mean, std)
    elif isinstance(layer, (nn.Linear, nn.BatchNorm2d)):
        torch.nn.init.normal_(layer.weight, mean, std)
        torch.nn.init.constant_(layer.bias, 0)


class Reshape(nn.Module):
    """Reshape layer.

    Parameters
    ----------
    shape: List[int]
        Dimensions after number of batches.

    """

    def __init__(self, shape: List[int]) -> None:
        """Initialize."""
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        """Perform forward.

        Parameters
        ----------
        x: Tensor
            Input batch.

        Returns
        -------
        Tensor:
            Preprocessed input batch.

        """
        return x.view(x.size(0), *self.shape)
