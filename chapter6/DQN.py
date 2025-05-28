# the model to help the agent learn to play the game using NN

import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        # inputshape[0] is the number of channels in the input image,
        # this goal to calculate q values through the network.
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # calculate the output shape after the conv layers

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        # create a dummy input to pass through the conv layers
        # shape is (C, H, W) expact channel first format
        o = self.conv(torch.zeros(1, *shape))
        # product of the dimensions of the output tensor #ex: 2, 3, 4 -> 2*3*4 = 24
        # o.size() returns a tuple of the dimensions of the output tensor
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        Forward pass through the network.
        x is expected to be a tensor of shape (batch_size, C, H, W)
        where C is the number of channels, H is the height and W is the width.

        The output is a tensor of shape (batch_size, n_actions) where n_actions"""
        # flatten the output of the conv layers base on the batch size
        # first we pass x through the conv layers, then using view we reshape the output to [batch_size, features] -1 mean flatten C x H x W
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
