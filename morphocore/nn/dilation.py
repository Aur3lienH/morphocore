from .morpho_module import MorphoModule
from morphocore.functional import dilation


class Dilation(MorphoModule):
    def __init__(self, in_channel : int, out_channel : int, kernel_shape : tuple, channel_merge_mode: str = "sum"):
        super().__init__(in_channel, out_channel, kernel_shape, channel_merge_mode)

    def forward(self, x):
        return dilation(x, self.weight, self.channel_merge_mode)
