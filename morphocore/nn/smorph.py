import torch
import torch.nn as nn
from .alpha_module import AlphaModule
from morphocore.functional import smorph


class SMorph(AlphaModule):

    """

    Module that approximate both erosion and dilation using an alpha parameter
    Formula : (Needs reflection)
    

    Behaviour of alpha : 
     - When alpha -> ∞ then Smorph tends to be a dilation 
     - When alpha -> -∞ then Smorph tends to be an erosion
     - When alpha is close to 0 -> then Smorph is something between an erosion and a dilation.

    """

    def __init__(self, in_channel: int, out_channel: int, kernel_shape: tuple, channel_merge_mode: str = "sum", init_alpha: float = 0.0):

        """
        Initialize Smorph Module
        
        Args:
            in_channel (int): Number of input channels
            out_channel (int): Number of output channels
            kernel_shape (tuple): Shape of the morphological kernel
        """

        super(SMorph, self).__init__(in_channel, out_channel, kernel_shape, channel_merge_mode, init_alpha)

    def forward(self, x : torch.Tensor):

        """
        Forward pass
        
        Args:
            x: Input tensor, shape : (batch, in_channels, height, width)

        Returns:
            Output tensor, shape : (batch, out_channels, height, width)
        """

        return smorph(x, self.weight, self.alpha, self.channel_merge_mode)