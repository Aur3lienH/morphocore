import torch
from torch import nn
from alpha_module import AlphaModule
from functional import emorph


class EMorph(AlphaModule):

    """
    Exact Morphological module that switch between erosion and dilation with a control parameter alpha.

    Formula : 
    The operation is Y(X) = sigmoid(alpha) * dilation(X, W) + (1.0 - sigmoid(alpha)) * erosion(X, W)
    where :
        - X is the input
        - W is the kernel
        - alpha is the control parameter between erosion and dilation

    Behaviour with different alpha : 
        - When alpha -> ∞ then Emorph tends to be a dilation 
        - When alpha -> -∞ then Emorph tends to be an erosion
        - When alpha is close to 0 -> then Emorph is something between an erosion and a dilation.
    """

    def __init__(self, in_channel : int, out_channel : int, kernel_shape : tuple):

        """
        Initialize Emorph Module
        
        Args:
            in_channel (int): Number of input channels
            out_channel (int): Number of output channels
            kernel_shape (tuple): Shape of the morphological kernel
        """

        super().__init__(in_channel, out_channel, kernel_shape)



    def forward(self, x):

        """
        Forward pass
        
        Args:
            x: Input tensor, shape : (batch, in_channels, height, width)

        Returns:
            Output tensor, shape : (batch, out_channels, height, width)
        """
        return emorph(x, self.weight, self.alpha)
        

