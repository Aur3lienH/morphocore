from .morpho_module import MorphoModule
from morphocore.functional import erosion


class Erosion(MorphoModule):
    """
    Erosion Module
    Perform an erosion on the input with the given weight
    Args:
        in_channel (int): Number of input channels
        out_channel (int): Number of output channels
        kernel_shape (tuple): Shape of the morphological kernel
        channel_merge_mode (str): Channel merge mode
    Returns:
        Output tensor, shape : (batch, out_channels, height, width)
    """
    def __init__(self, in_channel : int, out_channel : int, kernel_shape : tuple, channel_merge_mode: str = "sum"):
        super().__init__(in_channel, out_channel, kernel_shape, channel_merge_mode)

    def forward(self, x):
        return erosion(x, self.weight, self.channel_merge_mode)
