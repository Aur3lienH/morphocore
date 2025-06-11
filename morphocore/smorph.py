import torch
import torch.nn as nn
from alpha_module import AlphaModule
from modules.morpho_kernel import apply_kernel
from functional import smorph


class Smorph(AlphaModule):

    """

    Module that approximate both erosion and dilation using an alpha parameter
    Formula : (Needs reflection)
    

    Behaviour of alpha : 
     - When alpha -> ∞ then Smorph tends to be a dilation 
     - When alpha -> -∞ then Smorph tends to be an erosion
     - When alpha is close to 0 -> then Smorph is something between an erosion and a dilation.

    """

    def __init__(self, in_channel: int, out_channel: int, kernel_shape: tuple):

        """
        Initialize Smorph Module
        
        Args:
            in_channel (int): Number of input channels
            out_channel (int): Number of output channels
            kernel_shape (tuple): Shape of the morphological kernel
        """

        super(Smorph, self).__init__(in_channel, out_channel, kernel_shape)

    def forward(self, x : torch.Tensor):

        """
        Forward pass
        
        Args:
            x: Input tensor, shape : (batch, in_channels, height, width)

        Returns:
            Output tensor, shape : (batch, out_channels, height, width)
        """

        return smorph(x, self.weight, self.alpha)