import torch
from torch import nn


def initialize_weights(layer, mean=0.0, std=0.02):
    if isinstance(layer, (nn.Conv3d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(layer.weight, mean, std)
    elif isinstance(layer, (nn.Linear, nn.BatchNorm2d)):
        torch.nn.init.normal_(layer.weight, mean, std)
        torch.nn.init.constant_(layer.bias, 0)


class Reshape(nn.Module):
    def __init__(self, shape=[32, 1, 1]):
        super().__init__()
        self.shape = shape
        
    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, *self.shape)
