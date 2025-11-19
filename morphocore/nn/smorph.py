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

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, channel_merge_mode: str = "sum", init_alpha: float = 0.0, dtype: torch.dtype = torch.float32, init_beta: float = None):

        """
        Initialize Smorph Module
        
        Args:
            in_channel (int): Number of input channels
            out_channel (int): Number of output channels
            kernel_shape (tuple): Shape of the morphological kernel
        """

        super(SMorph, self).__init__(in_channels, out_channels, kernel_size, channel_merge_mode, init_alpha, dtype)
        if init_beta is not None:
            self.beta = nn.Parameter(torch.full((out_channels, in_channels), init_beta, dtype=dtype))
        else:
            self.beta = None

    def forward(self, x : torch.Tensor):

        """
        Forward pass
        
        Args:
            x: Input tensor, shape : (batch, in_channels, height, width)

        Returns:
            Output tensor, shape : (batch, out_channels, height, width)
        """

        return smorph(x=x, w=self.weight, alpha=self.alpha, beta=self.beta, channel_merge_mode=self.channel_merge_mode)