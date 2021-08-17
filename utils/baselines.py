
"""
Baselines for this task
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "17-08-2021"



import torch
from torch import nn

class Autoencoder(nn.Module):

    def __init__(self, data_module, n_hidden_channels: int, depthness: int):
        super(Autoencoder, self).__init__()
        #TODO: do autoencoder stuff
    def forward(self, x):
        pass

class Unet(nn.Module):
    def __init__(self, data_module, n_hidden_channels: int, depthness: int):
        super(Unet, self).__init__()
        #TODO: do unet stuff :)
    def forward(self, x):
        pass

