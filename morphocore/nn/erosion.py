from .morpho_module import MorphoModule
from morphocore.functional import erosion


class Erosion(MorphoModule):
    def __init__(self, in_channel : int, out_channel : int, kernel_shape : tuple):
        super().__init__(in_channel, out_channel, kernel_shape)

    def forward(self, x):
        return erosion(x, self.weight)
