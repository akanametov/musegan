import torch
from torch import nn
from .utils import Reshape


class TemporalNetwork(nn.Module):
    
    n_bars = 2
    
    def __init__(self,
                 z_dimension: int=32,
                 hid_channels: int=1024):
        super().__init__()        
        self.net = nn.Sequential(
            # input shape: (batch_size, z_dimension) 
            Reshape(shape=[z_dimension, 1, 1]),
            # output shape: (batch_size, z_dimension, 1, 1)
            nn.ConvTranspose2d(z_dimension, hid_channels,
                               kernel_size=(2, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, hid_channels, 2, 1)
            nn.ConvTranspose2d(hid_channels, z_dimension,
                               kernel_size=(self.n_bars-1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(z_dimension),
            nn.ReLU(inplace=True),
            # output shape: (batch_size, z_dimension, 1, 1)
            Reshape(shape=[z_dimension, self.n_bars])
        )

    def forward(self, x):
        fx = self.net(x)
        return fx
