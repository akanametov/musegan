"""Bar Generator."""

from torch import Tensor

import torch
from torch import nn
from .utils import Reshape


class BarGenerator(nn.Module):
    """Bar generator.

    Parameters
    ----------
    z_dimension: int, (default=32)
        Noise space dimension.
    hid_channels: int, (default=1024)
        Number of hidden channels.
    hid_features: int, (default=1024)
        Number of hidden features.
    out_channels: int, (default=1)
        Number of output channels.

    """

    n_steps_per_bar = 16
    n_pitches = 84

    def __init__(
        self,
        z_dimension: int = 32,
        hid_features: int = 1024,
        hid_channels: int = 512,
        out_channels: int = 1
    ) -> None:
        """Initialize."""
        super().__init__()
        self.net = nn.Sequential(
            # input shape: (batch_size, 4*z_dimension)
            nn.Linear(4 * z_dimension, hid_features),
            nn.BatchNorm1d(hid_features),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, hid_features)
            Reshape(shape=[hid_channels, hid_features // hid_channels, 1]),
            # output shape: (batch_size, hid_channels, hid_features//hid_channels, 1)
            nn.ConvTranspose2d(
                hid_channels,
                hid_channels,
                kernel_size=(2, 1),
                stride=(2, 1),
                padding=0,
            ),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, hid_channels, 2*hid_features//hid_channels, 1)
            nn.ConvTranspose2d(
                hid_channels,
                hid_channels // 2,
                kernel_size=(2, 1),
                stride=(2, 1),
                padding=0,
            ),
            nn.BatchNorm2d(hid_channels // 2),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, hid_channels//2, 4*hid_features//hid_channels, 1)
            nn.ConvTranspose2d(
                hid_channels // 2,
                hid_channels // 2,
                kernel_size=(2, 1),
                stride=(2, 1),
                padding=0,
            ),
            nn.BatchNorm2d(hid_channels // 2),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, hid_channels//2, 8*hid_features//hid_channels, 1)
            nn.ConvTranspose2d(
                hid_channels // 2,
                hid_channels // 2,
                kernel_size=(1, 7),
                stride=(1, 7),
                padding=0,
            ),
            nn.BatchNorm2d(hid_channels // 2),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, hid_channels//2, 8*hid_features//hid_channels, 7)
            nn.ConvTranspose2d(
                hid_channels // 2,
                out_channels,
                kernel_size=(1, 12),
                stride=(1, 12),
                padding=0,
            ),
            # output shape: (batch_size, out_channels, 8*hid_features//hid_channels, n_pitches)
            Reshape(shape=[1, 1, self.n_steps_per_bar, self.n_pitches])
            # output shape: (batch_size, out_channels, 1, n_steps_per_bar, n_pitches)
        )

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
        fx = self.net(x)
        return fx
