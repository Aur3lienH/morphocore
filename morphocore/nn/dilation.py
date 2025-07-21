from .morpho_module import MorphoModule
from morphocore.functional import dilation


class Dilation(MorphoModule):
    def __init__(self, in_channel : int, out_channel : int, kernel_shape : tuple):
        super().__init__(in_channel, out_channel, kernel_shape)

    def forward(self, x):
        return dilation(x, self.weight)
