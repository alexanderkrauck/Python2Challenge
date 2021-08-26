
"""
Baselines for this task
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "17-08-2021"



import torch
from torch import nn

import torch
from utils.basic_modules import *

class Autoencoder(nn.Module):

    def __init__(self, data_module, n_hidden_channels: int, depthness: int, **kwargs):
        super(Autoencoder, self).__init__()
        #TODO: do autoencoder stuff
    def forward(self, x):
        pass


class Unet(nn.Module):
    """UNet implementation

    This is taken from
    https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201

    I reduces the number of challels in the layers
    """

    def __init__(self, data_module, n_hidden_channels: int, depthness: int, **kwargs):
        super(Unet, self).__init__()

        self.e1 = EncoderBlock(1, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)

        self.b = ConvBlock(256, 512)

        self.d1 = DecoderBlock(512, 256)
        self.d2 = DecoderBlock(256, 128)
        self.d3 = DecoderBlock(128, 64)
        self.d4 = DecoderBlock(64, 32)

        self.outputs = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, x):

        x = torch.nn.functional.pad(x, (3,3,3,3), mode='constant', value=0)

        s1, p1 = self.e1(x)
        del x
        s2, p2 = self.e2(p1)
        del p1
        s3, p3 = self.e3(p2)
        del p2
        s4, p4 = self.e4(p3)
        del p3

        b = self.b(p4)
        del p4

        d1 = self.d1(b, s4)
        del b,s4
        d2 = self.d2(d1, s3)
        del d1, s3
        d3 = self.d3(d2, s2)
        del d2, s2
        d4 = self.d4(d3, s1)
        del d3, s1

        outputs = self.outputs(d4)
        del d4

        outputs = outputs[:,:,3:-3,3:-3]

        return outputs

