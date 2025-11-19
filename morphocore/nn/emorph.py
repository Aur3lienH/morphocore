import torch
from torch import nn
from .alpha_module import AlphaModule
from morphocore.functional import emorph


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

    def __init__(self, in_channels : int, out_channels : int, kernel_shape : tuple, channel_merge_mode: str = "sum", init_alpha: float = 0.0, dtype: torch.dtype = torch.float32, save_indices: bool = True):

        """
        Initialize Emorph Module
        
        Args:
            in_channel (int): Number of input channels
            out_channel (int): Number of output channels
            kernel_shape (tuple): Shape of the morphological kernel
        """

        super().__init__(in_channels, out_channels, kernel_shape, channel_merge_mode, init_alpha, dtype=dtype)
        self.w2 = nn.Parameter(torch.rand((out_channels,in_channels,*kernel_shape),dtype=dtype))
        self.save_indices = save_indices

    def forward(self, x):

        """
        Forward pass
        
        Args:
            x: Input tensor, shape : (batch, in_channels, height, width)

        Returns:
            Output tensor, shape : (batch, out_channels, height, width)
        """
        return emorph(x, self.weight, self.w2, self.alpha, self.channel_merge_mode, self.save_indices)