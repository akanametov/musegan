"""Muse critic."""

from torch import Tensor

import torch
from torch import nn


class MuseCritic(nn.Module):
    """Muse critic.

    Parameters
    ----------
    hid_channels: int, (default=128)
        Number of hidden channels.
    hid_features: int, (default=1024)
        Number of hidden features.
    out_channels: int, (default=1)
        Number of output channels.

    """

    n_tracks = 4
    n_bars = 2
    n_steps_per_bar = 16
    n_pitches = 84

    def __init__(
        self,
        hid_channels: int = 128,
        hid_features: int = 1024,
        out_features: int = 1,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.net = nn.Sequential(
            # input shape: (batch_size, n_tracks, n_bars, n_steps_per_bar, n_pitches)
            nn.Conv3d(self.n_tracks, hid_channels, (2, 1, 1), (1, 1, 1), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            # output shape: (batch_size, hid_channels, n_bars//2, n_steps_per_bar, n_pitches)
            nn.Conv3d(hid_channels, hid_channels, (self.n_bars - 1, 1, 1), (1, 1, 1), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            # output shape: (batch_size, hid_channels, n_bars//2, n_steps_per_bar, n_pitches)
            nn.Conv3d(hid_channels, hid_channels, (1, 1, 12), (1, 1, 12), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            # output shape: (batch_size, hid_channels, n_bars//2, n_steps_per_bar, n_pitches//12)
            nn.Conv3d(hid_channels, hid_channels, (1, 1, 7), (1, 1, 7), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            # output shape: (batch_size, hid_channels, n_bars//2, n_steps_per_bar//2, n_pitches//12)
            nn.Conv3d(hid_channels, hid_channels, (1, 2, 1), (1, 2, 1), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            # output shape: (batch_size, hid_channels, n_bars//2, n_steps_per_bar//4, n_pitches//12)
            nn.Conv3d(hid_channels, hid_channels, (1, 2, 1), (1, 2, 1), padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            # output shape: (batch_size, hid_channels, n_bars//2, n_steps_per_bar//4, n_pitches//12)
            nn.Conv3d(hid_channels, 2 * hid_channels, (1, 4, 1), (1, 2, 1), padding=(0, 1, 0)),
            nn.LeakyReLU(0.3, inplace=True),
            # output shape: (batch_size, hid_channels, n_bars//2, n_steps_per_bar//8, n_pitches//12)
            nn.Conv3d(2 * hid_channels, 4 * hid_channels, (1, 3, 1), (1, 2, 1), padding=(0, 1, 0)),
            nn.LeakyReLU(0.3, inplace=True),
            # output shape: (batch_size, hid_channels, n_bars//2, n_steps_per_bar//16, n_pitches//12)
            nn.Flatten(),
            nn.Linear(4 * hid_channels, hid_features),
            nn.LeakyReLU(0.3, inplace=True),
            # output shape: (batch_size, hid_features)
            nn.Linear(hid_features, out_features),
            # output shape: (batch_size, out_features)
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
