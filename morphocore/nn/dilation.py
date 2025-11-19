from .morpho_module import MorphoModule
from morphocore.functional import dilation
import torch


class Dilation(MorphoModule):
    """
    Dilation Module
    Perform a dilation on the input with the given weight
    Args:
        in_channel (int): Number of input channels
        out_channel (int): Number of output channels
        kernel_shape (tuple): Shape of the morphological kernel
        channel_merge_mode (str): Channel merge mode
    Returns:
        Output tensor, shape : (batch, out_channels, height, width)
    """
    def __init__(self, in_channels : int, out_channels : int, kernel_size : tuple, channel_merge_mode: str = "sum", dtype: torch.dtype = torch.float32, save_indices: bool = True):
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        self.save_indices = save_indices
        super().__init__(in_channels, out_channels, kernel_size, channel_merge_mode, dtype=dtype)

    def forward(self, x):
        return dilation(x, self.weight, self.channel_merge_mode, self.save_indices)
